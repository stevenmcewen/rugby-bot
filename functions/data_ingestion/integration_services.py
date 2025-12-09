from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Callable, Sequence
import os
from uuid import UUID, uuid4
from azure.storage.blob import BlobClient

from functions.data_ingestion.integration_helpers import (
    download_historical_results,
    scrape_values,
)
from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient

logger = get_logger(__name__)
settings = get_settings()


IngestionContext = dict[str, Any]


#### Pipeline core ####
class IngestionStep(Protocol):
    """
    A single ingestion step in the pipeline.

    Each step:
    - Receives a mutable `IngestionContext` dict.
    - Performs one responsibility.
    - Returns the (optionally) updated context.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext: ...


class IngestionPipeline:
    """
    Pipes & Filters style pipeline for ingestion.

    The pipeline is responsible for:
    - Running each step in sequence.
    - Passing an `IngestionContext` between steps.
    - Logging step execution.
    """

    def __init__(self, *steps: IngestionStep) -> None:
        self._steps: tuple[IngestionStep, ...] = steps

    def run(self, initial_context: IngestionContext) -> IngestionContext:
        context: IngestionContext = initial_context

        for step in self._steps:
            step_name = step.__class__.__name__
            logger.info("Running ingestion step: %s", step_name)
            context = step(context)

        return context


#### Step implementations ####
class LogIngestionEventStartStep:
    """
    Log the ingestion event start.

    Responsibility:
    - Insert a new row into ingestion_events to represent this ingestion batch.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info(
            "Logging ingestion event start for system_event_id=%s",
            context["system_event_id"],
        )

        sql_client: SqlClient = context["sql_client"]

        # Generate a new batch id (UUID) for this run.
        batch_id = uuid4()
        context["batch_id"] = batch_id

        try:
            # Use the current pipeline status so we don't overwrite a
            # previous failure (e.g. download/scrape step failed).
            status = context["status"]
            error_message = context.get("error_message", None)

            # Insert a new row into the ingestion_events table.
            ingestion_event_id = sql_client.start_ingestion_event(
                batch_id=batch_id,
                system_event_id=context["system_event_id"],
                container_name=context["raw_container_name"],
                integration_type=context["integration_type"],
                integration_provider=context["integration_provider"],
                status=status,
                blob_path=None,
                error_message=error_message,
            )
            # Update the context with the ingestion event id.
            context["ingestion_event_id"] = ingestion_event_id
        except Exception as e:
            logger.error("Error logging ingestion event start: %s", e)
            raise

        return context


class DownloadHistoricalDataStep:
    """
    Download historical rugby results from Kaggle.

    Accepts:
        integration_provider: The provider of the integration (e.g. "kaggle", "rugby365").

    Responsibility:
    - Download / read historical rugby results (e.g. Kaggle bootstrap).
    - Attach the full path of the downloaded data into the context.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info("Downloading historical rugby results via Kaggle.")

        # Populate integration metadata for this source.
        integration_provider = context["integration_provider"]

        try:
            local_file_path, integration_dataset = download_historical_results(integration_provider)
            context["local_integration_file_path"] = local_file_path
            context["integration_dataset"] = integration_dataset
            context["status"] = "started"
        except Exception as e:
            logger.error("Error downloading historical rugby results: %s", e)
            context["status"] = "failed"
            context["error_message"] = str(e)

        return context


class ScrapeResultsOrFixturesStep:
    """
    Placeholder step.

    Responsibility:
    - Scrape results or fixtures for a given provider (e.g. Rugby365).
    - Attach local file path and dataset metadata into the context.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        integration_type = context["integration_type"]
        provider = context["integration_provider"]

        logger.info(
            "Scraping %s data for provider=%s (placeholder implementation).",
            integration_type,
            provider,
        )

        # Populate integration metadata for this source.
        integration_provider = context["integration_provider"]
        integration_type = context["integration_type"]
        sql_client = context["sql_client"]

        try:
            # scrape the values for the given provider and type
            local_file_path, integration_dataset = scrape_values(integration_provider, integration_type, sql_client)
            # if there are no local file path or integration dataset, raise a ValueError
            context["local_integration_file_path"] = local_file_path
            context["integration_dataset"] = integration_dataset
            context["status"] = "started"
        except ValueError as e:
            logger.error("Error scraping values for provider=%s and type=%s: %s", provider, integration_type, e)
            context["status"] = "failed"
            context["error_message"] = str(e)
            raise
        except Exception as e:
            logger.error(
                "Unexpected error scraping values for provider=%s and type=%s: %s",
                provider,
                integration_type,
                e,
            )
            context["status"] = "failed"
            context["error_message"] = str(e)
            raise

        return context

class WriteRawSnapshotsToBlobStep:
    """
    Write raw snapshots to Blob Storage.

    Responsibility:
    - Persist raw snapshots (from previous steps) into Blob Storage.
    - Use the configured raw container name from settings.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        # If a previous step has already marked the context as failed,
        # skip any blob work to avoid masking the original error.
        if context.get("status") == "failed":
            logger.warning(
                "Skipping blob write because ingestion context is already in a failed state "
                "(system_event_id=%s, ingestion_event_id=%s)",
                context.get("system_event_id"),
                context.get("ingestion_event_id"),
            )
            return context

        logger.info(
            "Writing raw snapshots to Blob Storage (container=%s).",
            settings.raw_container_name,
        )

        try:
            local_path = context["local_integration_file_path"]

            blob_name = (
                f"{context['integration_type']}/"
                f"{context['integration_provider']}/"
                f"{context['integration_dataset']}.csv"
            )

            blob_client = BlobClient.from_connection_string(
                conn_str=settings.storage_connection,
                container_name=context["raw_container_name"],
                blob_name=blob_name,
            )

            # Upload the actual file contents, not just the local path string.
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            context["blob_snapshot_uri"] = blob_name
            # mark success if not already failed by earlier steps
            if context.get("status") != "failed":
                context["status"] = "ingested"

        except Exception as e:
            logger.error("Error writing raw snapshots to Blob Storage: %s", e)
            context["status"] = "failed"
            context["error_message"] = str(e)

        return context


class LogIngestionEventCompleteStep:
    """
    Log the ingestion event completion.

    Responsibility:
    - Update the ingestion_events row with final status and blob path (if any).
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info(
            "Logging ingestion event end for ingestion_event_id=%s",
            context["ingestion_event_id"],
        )

        sql_client: SqlClient = context["sql_client"]
        ingestion_event_id = context["ingestion_event_id"]
        status = context["status"]

        if status == "failed":
            error_message = context.get("error_message", "Unknown error")
            logger.error("Ingestion event failed: %s", error_message)
            sql_client.update_ingestion_event(
                ingestion_event_id=ingestion_event_id,
                status="failed",
                blob_path=context.get("blob_snapshot_uri"),
                error_message=error_message,
            )
            # propagate the error so the function run is marked as failure
            raise Exception(error_message)
        else:
            logger.info(
                "Ingestion event completed successfully for ingestion_event_id=%s",
                context["ingestion_event_id"],
            )
            sql_client.update_ingestion_event(
                ingestion_event_id=ingestion_event_id,
                status="ingested",
                blob_path=context.get("blob_snapshot_uri"),
                error_message=None,
            )

        return context


#### Pipeline registry / factory ####
PipelineFactory = Callable[[], Sequence[IngestionStep]]
PIPELINE_REGISTRY: dict[str, PipelineFactory] = {}


def register_pipeline(name: str, factory: PipelineFactory) -> None:
    """
    Register a named ingestion pipeline.

    `name` is how callers (or Azure Functions) will refer to this pipeline,
    e.g. 'historical', 'rugby365_results', 'rugby365_fixtures'.
    """
    if name in PIPELINE_REGISTRY:
        raise ValueError(f"Pipeline {name!r} is already registered.")
    PIPELINE_REGISTRY[name] = factory


def build_pipeline_for(name: str) -> IngestionPipeline:
    """
    Build an IngestionPipeline instance for the given registered name.
    """
    try:
        factory = PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"No pipeline registered under name {name!r}") from exc

    steps = factory()
    return IngestionPipeline(*steps)


def run_ingestion(
    pipeline_name: str,
    sql_client: SqlClient,
    system_event_id: UUID,
    integration_type: str,
    integration_provider: str,
) -> None:
    """
    Generic ingestion entrypoint.

    Args:
        pipeline_name: Name of the pipeline registered in PIPELINE_REGISTRY.
        sql_client: SqlClient for logging ingestion_events.
        system_event_id: Parent system_events.id.
        integration_type: The type of integration (e.g. "results", "fixtures").
        integration_provider: The provider of the integration (e.g. "kaggle", "rugby365").
    """
    logger.info(
        "Running ingestion pipeline '%s' into bronze layer (container=%s)",
        pipeline_name,
        settings.raw_container_name,
    )

    pipeline = build_pipeline_for(pipeline_name)

    context: IngestionContext = {
        "raw_container_name": settings.raw_container_name,
        "sql_client": sql_client,
        "system_event_id": system_event_id,
        "status": "started",
        "integration_type": integration_type,
        "integration_provider": integration_provider,
    }

    pipeline.run(context)


#### Concrete pipeline registrations ####
def _historical_pipeline_factory() -> tuple[IngestionStep, ...]:
    """
    Pipeline for historical Kaggle-based ingestion:
        Download → LogStart → WriteToBlob → LogComplete
    """
    return (
        DownloadHistoricalDataStep(),
        LogIngestionEventStartStep(),
        WriteRawSnapshotsToBlobStep(),
        LogIngestionEventCompleteStep(),
    )


def _rugby365_results_pipeline_factory() -> tuple[IngestionStep, ...]:
    """
    Pipeline for Rugby365 results:
        ScrapeResults → LogStart → WriteToBlob → LogComplete
    """
    return (
        ScrapeResultsOrFixturesStep(),
        LogIngestionEventStartStep(),
        WriteRawSnapshotsToBlobStep(),
        LogIngestionEventCompleteStep(),
    )


def _rugby365_fixtures_pipeline_factory() -> tuple[IngestionStep, ...]:
    """
    Pipeline for Rugby365 fixtures:
        ScrapeFixtures → LogStart → WriteToBlob → LogComplete
    """
    return (
        ScrapeResultsOrFixturesStep(),
        LogIngestionEventStartStep(),
        WriteRawSnapshotsToBlobStep(),
        LogIngestionEventCompleteStep(),
    )


# Register pipelines at import time
register_pipeline("kaggle_historical_results", _historical_pipeline_factory)
register_pipeline("rugby365_results", _rugby365_results_pipeline_factory)
register_pipeline("rugby365_fixtures", _rugby365_fixtures_pipeline_factory)


#### Public ingestion functions used by Azure Functions ####
def ingest_historical_kaggle_results(sql_client: SqlClient, system_event_id: UUID) -> None:
    """
    Orchestrate ingestion of historical rugby results data into the bronze layer.
    Intended to be run infrequently (e.g. one-off or on-demand).
    """
    run_ingestion(
        pipeline_name="kaggle_historical_results",
        sql_client=sql_client,
        system_event_id=system_event_id,
        integration_type="results",
        integration_provider="kaggle",
    )


def ingest_rugby365_results(sql_client: SqlClient, system_event_id: UUID) -> None:
    """
    Orchestrate ingestion of Rugby365 results data into the bronze layer.
    Intended to be run on a schedule (e.g. daily).
    """
    run_ingestion(
        pipeline_name="rugby365_results",
        sql_client=sql_client,
        system_event_id=system_event_id,
        integration_type="results",
        integration_provider="rugby365",
    )


def ingest_rugby365_fixtures(sql_client: SqlClient, system_event_id: UUID) -> None:
    """
    Orchestrate ingestion of Rugby365 fixtures data into the bronze layer.
    Intended to be run on a schedule (e.g. daily).
    """
    run_ingestion(
        pipeline_name="rugby365_fixtures",
        sql_client=sql_client,
        system_event_id=system_event_id,
        integration_type="fixtures",
        integration_provider="rugby365",
    )
