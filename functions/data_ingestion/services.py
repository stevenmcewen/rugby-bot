from __future__ import annotations

from typing import Any, Protocol

from functions.config.settings import get_settings
from functions.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


IngestionContext = dict[str, Any]


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
    Simple Pipes & Filters style pipeline for ingestion.

    The pipeline is responsible for:
    - Running each step in sequence.
    - Passing an `IngestionContext` between steps.
    - Logging step execution.
    """

    def __init__(self, *steps: IngestionStep) -> None:
        self._steps: tuple[IngestionStep, ...] = steps

    def run(self, initial_context: IngestionContext | None = None) -> IngestionContext:
        context: IngestionContext = initial_context or {}

        for step in self._steps:
            step_name = step.__class__.__name__
            logger.info("Running ingestion step: %s", step_name)
            context = step(context)

        return context


class DownloadHistoricalDataStep:
    """
    Placeholder step.

    Responsibility:
    - Download / read historical rugby data (e.g. Kaggle bootstrap).
    - Record where the raw data was stored in the context.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info("Downloading historical rugby data (placeholder implementation).")

        # TODO: implement concrete download logic and populate this list.
        context["historical_data_paths"] = []

        return context


class ScrapeRugby365ResultsStep:
    """
    Placeholder step.

    Responsibility:
    - Scrape Rugby365 results.
    - Attach scraped data or locations into the context.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info("Scraping Rugby365 results (placeholder implementation).")

        # TODO: implement concrete scraping logic and populate this list.
        context["rugby365_results_paths"] = []

        return context


class ScrapeRugby365FixturesStep:
    """
    Placeholder step.

    Responsibility:
    - Scrape Rugby365 fixtures.
    - Attach scraped data or locations into the context.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info("Scraping Rugby365 fixtures (placeholder implementation).")

        # TODO: implement concrete scraping logic and populate this list.
        context["rugby365_fixtures_paths"] = []

        return context


class WriteRawSnapshotsToBlobStep:
    """
    Placeholder step.

    Responsibility:
    - Persist raw snapshots (from previous steps) into Blob Storage.
    - Use the configured raw container name from settings.
    """

    def __call__(self, context: IngestionContext) -> IngestionContext:
        logger.info(
            "Writing raw snapshots to Blob Storage (placeholder implementation, container=%s).",
            settings.raw_container_name,
        )

        # TODO: implement concrete upload logic to Blob Storage.
        # Example shape of the context field for downstream steps:
        context["blob_snapshot_uris"] = []

        return context


def ingest_historical_results() -> None:
    """
    Orchestrate ingestion of historical rugby results data into the bronze layer.

    This is intended to be run infrequently (e.g. one-off or on-demand).
    """

    logger.info(
        "Ingesting *historical* rugby results into bronze layer (container=%s)",
        settings.raw_container_name,
    )

    pipeline = IngestionPipeline(
        DownloadHistoricalDataStep(),
        WriteRawSnapshotsToBlobStep(),
    )

    # Initial context can carry any static configuration or request metadata.
    initial_context: IngestionContext = {
        "raw_container_name": settings.raw_container_name,
        "mode": "historical",
    }

    pipeline.run(initial_context)


def ingest_rugby365_results() -> None:
    """
    Orchestrate ingestion of Rugby365 results data into the bronze layer.

    This is intended to be run on a schedule (e.g. daily).
    """

    logger.info(
        "Ingesting Rugby365 results into bronze layer (container=%s)",
        settings.raw_container_name,
    )

    pipeline = IngestionPipeline(
        ScrapeRugby365ResultsStep(),
        WriteRawSnapshotsToBlobStep(),
    )

    initial_context: IngestionContext = {
        "raw_container_name": settings.raw_container_name,
        "mode": "results",
    }

    pipeline.run(initial_context)


def ingest_rugby365_fixtures() -> None:
    """
    Orchestrate ingestion of Rugby365 fixtures data into the bronze layer.

    This is intended to be run on a schedule (e.g. daily).
    """

    logger.info(
        "Ingesting Rugby365 fixtures into bronze layer (container=%s)",
        settings.raw_container_name,
    )

    pipeline = IngestionPipeline(
        ScrapeRugby365FixturesStep(),
        WriteRawSnapshotsToBlobStep(),
    )

    initial_context: IngestionContext = {
        "raw_container_name": settings.raw_container_name,
        "mode": "fixtures",
    }

    pipeline.run(initial_context)


def ingest_rugby_data() -> None:
    """
    Backwards-compatible generic ingestion entrypoint.

    Currently runs the scheduled-style ingests (results + fixtures),
    but does NOT run the historical loader.

    Prefer calling the more specific functions:
    - ingest_historical_results
    - ingest_rugby365_results
    - ingest_rugby365_fixtures
    """

    logger.info("ingest_rugby_data() called; delegating to results + fixtures ingests.")

    ingest_rugby365_results()
    ingest_rugby365_fixtures()

