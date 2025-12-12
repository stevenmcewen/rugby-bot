from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, Sequence, Any
from uuid import UUID

from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient
from functions.data_preprocessing.preprocessing_pipelines import run_preprocessing_pipeline


logger = get_logger(__name__)
settings = get_settings()

PreprocessingContext = dict[str, Any]

## Pipeline step interface ##
class PreprocessingStep(Protocol):
    """
    Single responsibility step in the preprocessing pipeline.

    each step:
    - accepts a mutable `PreprocessingContext`
    - performs exactly one responsibility
    - returns the updated context
    """

    def __call__(self, context: PreprocessingContext) -> PreprocessingContext:
        ...


class PreprocessingPipeline:
    """
    Light‑weight orchestration of a sequence of `PreprocessingStep`s.
    The pipeline is responsible for:
    - Running each step in sequence.
    - Passing an `PreprocessingContext` between steps.
    - Logging step execution.
    """

    def __init__(self, *steps: PreprocessingStep) -> None:
        self._steps: tuple[PreprocessingStep, ...] = steps

    def run(self, initial_context: PreprocessingContext) -> PreprocessingContext:
        context = initial_context
        for step in self._steps:
            step_name = step.__class__.__name__
            logger.info("Running preprocessing step: %s", step_name)
            context = step(context)
        return context


### Domain shells ###
@dataclass(frozen=True)
class IngestionSource:
    """
    Represents a *source* that has been successfully ingested into bronze
    and is now eligible for preprocessing.

    This mirrors a row in `dbo.ingestion_events`.
    """

    id: UUID
    batch_id: UUID
    system_event_id: UUID
    integration_type: str
    integration_provider: str
    container_name: str
    blob_path: str


@dataclass(frozen=True)
class PreprocessingPlan:
    """
    A logical plan describing *what* preprocessing needs to run for one
    ingestion source.

    This is where we model the “source → one or many targets” mapping,
    typically backed by a dedicated mapping table, e.g.:

        dbo.preprocessing_source_targets
            (source_provider, source_type, target_table, pipeline_name, …)
    """

    batch_id: UUID
    system_event_id: UUID
    integration_type: str
    integration_provider: str
    container_name: str
    blob_path: str
    target_table: str
    pipeline_name: str


@dataclass()
class PreprocessingEvent:
    """
    A preprocessing event is a row in the dbo.preprocessing_events table.
    """
    id: UUID
    batch_id: UUID
    system_event_id: UUID
    integration_type: str
    integration_provider: str
    container_name: str
    blob_path: str
    target_table: str
    pipeline_name: str
    status: str
    error_message: str | None

### Step implementations ###
class SelectIngestedSourcesStep:
    """
    This step selects the ingestion_sources that are ready for preprocessing.
    It gets the ingestion_events with status='ingested' and then creates the ingestion_sources.
    It then adds the ingestion_sources to the context.
    """

    def __call__(self, context: PreprocessingContext) -> PreprocessingContext:
        logger.info("Selecting ingestion_events with status='ingested'.")

        try:
            sql_client: SqlClient = context["sql_client"]
            ingestion_events = sql_client.get_ingestion_events_by_status(status="ingested")

            ingestion_sources = []
            # for each ingestion event, create an ingestion source
            for ingestion_event in ingestion_events:
                ingestion_source = IngestionSource(
                    id=ingestion_event.id,
                    batch_id=ingestion_event.batch_id,
                    system_event_id=ingestion_event.system_event_id,
                    integration_type=ingestion_event.integration_type,
                    integration_provider=ingestion_event.integration_provider,
                    container_name=ingestion_event.container_name,
                    blob_path=ingestion_event.blob_path
                )
                ingestion_sources.append(ingestion_source)
            context["ingestion_sources"] = ingestion_sources
        except Exception as e:
            logger.error("Error getting ingestion_sources with status='ingested': %s", e)
            raise
        return context


class BuildPreprocessingPlansStep:
    """
    This step builds the preprocessing plans from the source→target mapping.
    It gets the source_target_mappings from the database (dbo.preprocessing_source_target_mappings) and then creates the preprocessing plans.
    It then adds the preprocessing plans to the context for the next step.
    """

    def __call__(self, context: PreprocessingContext) -> PreprocessingContext:
        logger.info("Building preprocessing plans from source to target mapping.")

        try:
            sql_client: SqlClient = context["sql_client"]
            ingestion_sources = context["ingestion_sources"]
            preprocessing_plans = []
            # loop through each ingestion source)
            for ingestion_source in ingestion_sources:
                # get the source_target_mappings for the ingestion source from the database
                source_target_mappings = sql_client.get_source_target_mapping(source_provider=ingestion_source.integration_provider, source_type=ingestion_source.integration_type)
                for source_target_mapping in source_target_mappings:
                    # create a preprocessing plan for the ingestion source and source_target_mapping
                    preprocessing_plan = PreprocessingPlan(
                        batch_id=ingestion_source.batch_id,
                        system_event_id=ingestion_source.system_event_id,
                        integration_type=ingestion_source.integration_type,
                        integration_provider=ingestion_source.integration_provider,
                        container_name=ingestion_source.container_name,
                        blob_path=ingestion_source.blob_path,
                        target_table=source_target_mapping["target_table"],
                        pipeline_name=source_target_mapping["pipeline_name"],
                    )
                    preprocessing_plans.append(preprocessing_plan)
            context["preprocessing_plans"] = preprocessing_plans
        except Exception as e:
            logger.error("Error building preprocessing plans: %s", e)
            raise
        return context


class PersistPreprocessingEventsStep:
    """
    This step persists the preprocessing plans into the database (dbo.preprocessing_events).
    It creates a preprocessing event for each preprocessing plan and adds the preprocessing event ids to the context for the next step.
    """

    def __call__(self, context: PreprocessingContext) -> PreprocessingContext:
        logger.info("Persisting preprocessing_events for planned work (shell only).")

        try:
            sql_client: SqlClient = context["sql_client"]
            preprocessing_plans = context["preprocessing_plans"]
            preprocessing_events = []
            for preprocessing_plan in preprocessing_plans:
                preprocessing_event_details = sql_client.create_preprocessing_event(preprocessing_plan=preprocessing_plan)
                preprocessing_event = PreprocessingEvent(
                    id=preprocessing_event_details["id"],
                    batch_id=preprocessing_event_details["batch_id"],
                    system_event_id=preprocessing_event_details["system_event_id"],
                    integration_type=preprocessing_event_details["integration_type"],
                    integration_provider=preprocessing_event_details["integration_provider"],
                    container_name=preprocessing_event_details["container_name"],
                    blob_path=preprocessing_event_details["blob_path"],
                    target_table=preprocessing_event_details["target_table"],
                    pipeline_name=preprocessing_event_details["pipeline_name"],
                    status=preprocessing_event_details["status"],
                    error_message=preprocessing_event_details["error_message"],
                )
                preprocessing_events.append(preprocessing_event)
            context["preprocessing_events"] = preprocessing_events
        except Exception as e:
            logger.error("Error persisting preprocessing events: %s", e)
            raise
        return context


class RunPreprocessingPipelinesStep:
    """
    Step 4: Run the preprocessing pipelines for each preprocessing event.
    It runs the preprocessing pipelines for each preprocessing event and updates the preprocessing event status according to the pipeline results.
    It then updates the ingestion event status to 'preprocessed' if the preprocessing event status is 'succeeded' for all the preprocessing events for the ingestion event.

    """

    def __call__(self, context: PreprocessingContext) -> PreprocessingContext:
        logger.info("Running concrete preprocessing pipelines.")
        preprocessing_events = context["preprocessing_events"]
        ingestion_sources = context["ingestion_sources"]
        try:
            sql_client: SqlClient = context["sql_client"]
            for preprocessing_event in preprocessing_events:
                run_preprocessing_pipeline(preprocessing_event, sql_client)
                if preprocessing_event.status == "succeeded":
                    sql_client.update_preprocessing_event(preprocessing_event_id=preprocessing_event.id, status="succeeded")
                else:
                    sql_client.update_preprocessing_event(preprocessing_event_id=preprocessing_event.id, status="failed", error_message=preprocessing_event.error_message)
            # update the ingestion event status to 'preprocessed' if the preprocessing event status is 'succeeded' for all the preprocessing events for the ingestion event
            for ingestion_source in ingestion_sources:
                preprocessing_events = sql_client.get_preprocessing_events_by_batch_id(batch_id=ingestion_source.batch_id)
                # SQLAlchemy returns row objects here, so access columns by attribute
                if all(preprocessing_event.status == "succeeded" for preprocessing_event in preprocessing_events):
                    sql_client.update_ingestion_event(ingestion_event_id=ingestion_source.id, status="preprocessed", error_message=None)
                else:
                    sql_client.update_ingestion_event(ingestion_event_id=ingestion_source.id, status="failed to preprocess", error_message="Preprocessing failed for some preprocessing events for this batch")
        except Exception as e:
            logger.error("Error running preprocessing pipelines: %s", e)
            raise
        return context

### Pipeline registry / factory ###
PreprocessingPipelineFactory = Callable[[], Sequence[PreprocessingStep]]
PREPROCESSING_PIPELINE_REGISTRY: dict[str, PreprocessingPipelineFactory] = {}


def register_preprocessing_pipeline(name: str, factory: PreprocessingPipelineFactory) -> None:
    """
    Register a named orchestration pipeline for the silver‑layer build.

    Example names:
      - "default_bronze_to_silver"
      - "backfill_only"
    """
    if name in PREPROCESSING_PIPELINE_REGISTRY:
        raise ValueError(f"Preprocessing pipeline {name!r} is already registered.")
    PREPROCESSING_PIPELINE_REGISTRY[name] = factory


def build_preprocessing_pipeline_for(name: str) -> PreprocessingPipeline:
    """
    Resolve a registered orchestration pipeline instance.
    """
    try:
        factory = PREPROCESSING_PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"No preprocessing pipeline registered under name {name!r}") from exc

    steps = factory()
    return PreprocessingPipeline(*steps)

### factory for the default preprocessing pipeline ###
def _default_preprocessing_factory() -> tuple[PreprocessingStep, ...]:
    """
    Default end‑to‑end orchestration from bronze → silver:
        1. Discover eligible sources
        2. Map to targets/pipelines
        3. Create preprocessing_events rows
        4. Run concrete preprocessing work

    Each responsibility is encapsulated as a step so we can extend or
    reorder later without rewriting the orchestration.
    """

    return (
        SelectIngestedSourcesStep(),
        BuildPreprocessingPlansStep(),
        PersistPreprocessingEventsStep(),
        RunPreprocessingPipelinesStep(),
    )


# Register the default orchestration pipeline at import time.
register_preprocessing_pipeline("default_preprocessing", _default_preprocessing_factory)


### Public entrypoint orchestrating preprocessing from bronze to silver ###
def orchestrate_preprocessing(
    *,
    sql_client: SqlClient,
    system_event_id: UUID,
    pipeline_name: str = "default_preprocessing",
) -> None:
    """
    High‑level orchestration entrypoint for the preprocessing phase.

    Current behaviour is deliberately a no‑op shell:
    it wires together the design pattern and logging without yet
    touching SQL or Blob, so you can evolve the detailed steps
    incrementally without breaking the running Function.
    """

    logger.info("Starting preprocessing orchestration with pipeline=%s", pipeline_name)

    pipeline = build_preprocessing_pipeline_for(pipeline_name)

    context: PreprocessingContext = {
        "sql_client": sql_client,
        "system_event_id": system_event_id,
        "status": "started",
        "error_message": None,
    }

    pipeline.run(context)
