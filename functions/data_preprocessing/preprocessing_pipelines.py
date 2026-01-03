
from __future__ import annotations

from typing import Callable, Protocol, TYPE_CHECKING

from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient

if TYPE_CHECKING:
    from functions.data_preprocessing.preprocessing_services import PreprocessingEvent

from functions.data_preprocessing.preprocessing_helpers import (
    get_source_data, 
    get_source_schema, 
    get_target_schema, 
    validate_source_data,
    validate_transformed_data,
    transform_kaggle_historical_data_to_international_results,
    transform_rugby365_results_data_to_international_results,
    transform_rugby365_fixtures_data_to_international_fixtures,
    transform_international_results_to_model_ready_data,
    truncate_target_table,
    write_data_to_target_table
)

"""
Helper functions for the preprocessing layer.

This module is responsible for the concrete preprocessing pipelines structures.
The logic for the preprocessing pipelines is defined in the preprocessing_helpers module.
The preprocessing pipelines are registered in the PREPROCESSING_HANDLER_REGISTRY dictionary.
The preprocessing pipelines are executed in the run_preprocessing_pipeline function.

Should you require a new preprocessing pipeline, you can add it to the PREPROCESSING_HANDLER_REGISTRY dictionary and define the logic in the
pipeline functions section at the top of this file.
The pipeline function should be a function that takes a preprocessing event and a sql client and returns a None.
The building blocks of a preprocessing pipeline can be found in the preprocessing_helpers module.
"""

logger = get_logger(__name__)
settings = get_settings()

### Pipeline functions ###
def historical_kaggle_international_results_preprocessing_pipeline(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """
    Preprocessing pipeline for historical Kaggle data.
    """
    try:
        # get the source data
        source_data = get_source_data(preprocessing_event, sql_client)
        # if there are no source rows, return
        if source_data.empty:
            logger.warning(
                "No source rows for preprocessing_event=%s; skipping preprocessing.",
                preprocessing_event.id,
            )
            return
        # get the source schema
        source_schema = get_source_schema(preprocessing_event, sql_client)
        # get the target schema
        target_schema = get_target_schema(preprocessing_event, sql_client)
        # validate the source data
        validate_source_data(source_data, source_schema)
        # transform the data
        transformed_data = transform_kaggle_historical_data_to_international_results(source_data, preprocessing_event, sql_client)
        # if there are no transformed rows, treat as a no-op (nothing new to write)
        if transformed_data.empty:
            logger.warning(
                "No transformed Kaggle rows for preprocessing_event=%s; skipping write.",
                preprocessing_event.id,
            )
            return
        # validate the transformed data
        validate_transformed_data(transformed_data, target_schema)
        # write the data to the target table
        write_data_to_target_table(transformed_data, preprocessing_event, sql_client)
    except Exception as e:
        logger.error("Error running historical Kaggle preprocessing pipeline for event %s: %s", preprocessing_event.id, e)
        raise

def rugby365_international_results_preprocessing_pipeline(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """
    Preprocessing pipeline for Rugby365 results data.
    """
    try:
        # get the source data
        source_data = get_source_data(preprocessing_event, sql_client)
        # if there are no source rows, return
        if source_data.empty:
            logger.warning(
                "No source rows for preprocessing_event=%s; skipping preprocessing.",
                preprocessing_event.id,
            )
            return
        # get the source schema
        source_schema = get_source_schema(preprocessing_event, sql_client)
        # get the target schema
        target_schema = get_target_schema(preprocessing_event, sql_client)
        # validate the source data
        validate_source_data(source_data, source_schema)
        # transform the data
        transformed_data = transform_rugby365_results_data_to_international_results(source_data, preprocessing_event, sql_client)
        # if there are no transformed rows, treat as a no-op (nothing new to write)
        if transformed_data.empty:
            logger.warning(
                "No transformed Rugby365 results rows for preprocessing_event=%s; skipping write.",
                preprocessing_event.id,
            )
            return
        # validate the transformed data
        validate_transformed_data(transformed_data, target_schema)
        # write the data to the target table
        write_data_to_target_table(transformed_data, preprocessing_event, sql_client)
    except Exception as e:
        logger.error("Error running Rugby365 results preprocessing pipeline for event %s: %s", preprocessing_event.id, e)
        raise

def rugby365_international_fixtures_preprocessing_pipeline(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """
    Preprocessing pipeline for Rugby365 fixtures data.
    """
    try:
        # get the source data
        source_data = get_source_data(preprocessing_event, sql_client)
        # if there are no source rows, return
        if source_data.empty:
            logger.warning(
                "No source rows for preprocessing_event=%s; skipping preprocessing.",
                preprocessing_event.id,
            )
            return
        # get the source schema
        source_schema = get_source_schema(preprocessing_event, sql_client)
        # get the target schema
        target_schema = get_target_schema(preprocessing_event, sql_client)
        # validate the source data
        validate_source_data(source_data, source_schema)
        # transform the data
        transformed_data = transform_rugby365_fixtures_data_to_international_fixtures(source_data, preprocessing_event, sql_client)
        # if there are no transformed rows, treat as a no-op (nothing new to write)
        if transformed_data.empty:
            logger.warning(
                "No transformed Rugby365 fixtures rows for preprocessing_event=%s; skipping write.",
                preprocessing_event.id,
            )
            return
        # validate the transformed data
        validate_transformed_data(transformed_data, target_schema)
        # write the data to the target table
        write_data_to_target_table(transformed_data, preprocessing_event, sql_client)
    except Exception as e:
        logger.error("Error running Rugby365 results preprocessing pipeline for event %s: %s", preprocessing_event.id, e)
        raise


def international_results_to_model_ready_data_preprocessing_pipeline(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """
    Preprocessing pipeline for InternationalMatchResults -> InternationalMatchResultsModelData.

    This is a "second layer" transformation:
      - Reads the silver facts table from SQL
      - Builds leakage-safe pre-kickoff features + target (home_win)
      - Rebuilds the model table deterministically
    """
    try:
        # get the source data
        source_data = get_source_data(preprocessing_event, sql_client)
        # if there are no source rows, return
        if source_data.empty:
            logger.warning(
                "No source rows for preprocessing_event=%s; skipping preprocessing.",
                preprocessing_event.id,
            )
            return
        # get the source schema
        source_schema = get_source_schema(preprocessing_event, sql_client)
        # get the target schema
        target_schema = get_target_schema(preprocessing_event, sql_client)
        # validate the source data
        validate_source_data(source_data, source_schema)
        # transform the data
        transformed_data = transform_international_results_to_model_ready_data(source_data, preprocessing_event, sql_client)
        # if there are no transformed rows, treat as a no-op (nothing new to write)
        if transformed_data.empty:
            logger.warning(
                "No transformed international results rows for preprocessing_event=%s; skipping write.",
                preprocessing_event.id,
            )
            return
        # validate the transformed data
        validate_transformed_data(transformed_data, target_schema)
        # truncate the target table
        truncate_target_table(preprocessing_event, sql_client)
        # write the data to the target table
        write_data_to_target_table(transformed_data, preprocessing_event, sql_client)
    except Exception as e:
        logger.error("Error building model-ready data for event %s: %s", preprocessing_event.id, e)
        raise

#### Preprocessing handler interface ####
class PreprocessingHandler(Protocol):
    """
    Signature for a concrete preprocessing function.

    Implementations are responsible for:
      - Reading the bronze data for `preprocessing_event`.
      - Performing any transformations / validation.
      - Writing to the silver target table (`preprocessing_event.target_table`).
    """

    def __call__(self, preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None: ...


# Registry of concrete preprocessing handlers, keyed by `pipeline_name`.
PREPROCESSING_HANDLER_REGISTRY: dict[str, PreprocessingHandler] = {
    "historical_kaggle_international_results_preprocessing": historical_kaggle_international_results_preprocessing_pipeline,
    "rugby365_international_results_preprocessing": rugby365_international_results_preprocessing_pipeline,
    "rugby365_international_fixtures_preprocessing": rugby365_international_fixtures_preprocessing_pipeline,
    "international_results_to_model_ready_data_preprocessing": international_results_to_model_ready_data_preprocessing_pipeline,
}


def register_preprocessing_pipeline(name: str, handler: PreprocessingHandler) -> None:
    """
    Register a concrete preprocessing function under a logical name.

    Typical usage from a module that implements a concrete pipeline:

        from functions.data_preprocessing.preprocessing_helpers import (
            register_preprocessing_pipeline,
        )

        def run_results_to_matches(event: PreprocessingEvent, sql: SqlClient) -> None:
            ...

        register_preprocessing_pipeline("results_to_matches", run_results_to_matches)
    """
    if name in PREPROCESSING_HANDLER_REGISTRY:
        raise ValueError(f"Preprocessing handler {name!r} is already registered.")
    PREPROCESSING_HANDLER_REGISTRY[name] = handler


def _resolve_preprocessing_handler(name: str) -> PreprocessingHandler:
    """
    Resolve the concrete handler for a given pipeline name.
    """
    try:
        return PREPROCESSING_HANDLER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"No preprocessing handler registered under name {name!r}") from exc


### main entrypoint for one preprocessing event ###
def run_preprocessing_pipeline(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """
    Run the concrete preprocessing work for a single `PreprocessingEvent`.

    Behaviour:
      - Marks the event as `running` in the database.
      - Looks up the concrete handler by `preprocessing_event.pipeline_name`.
      - Executes the handler.
      - Updates the in‑memory event status to `succeeded`/`failed` and
        populates `error_message` accordingly.

    The caller (`RunPreprocessingPipelinesStep`) is responsible for
    persisting the final status back to SQL.
    """
    logger.info(
        "Starting preprocessing pipeline %r for event_id=%s",
        getattr(preprocessing_event, "pipeline_name", None),
        getattr(preprocessing_event, "id", None),
    )

    # Mark as running up‑front so failures later are visible.
    preprocessing_event.status = "running"
    sql_client.update_preprocessing_event(
        preprocessing_event_id=preprocessing_event.id,
        status="running",
    )

    # Resolve the appropriate concrete handler for this event.
    try:
        handler = _resolve_preprocessing_handler(preprocessing_event.pipeline_name)
    except Exception as exc:
        logger.error(
            "Failed to resolve preprocessing handler for pipeline_name=%r: %s",
            getattr(preprocessing_event, "pipeline_name", None),
            exc,
        )
        preprocessing_event.status = "failed"
        preprocessing_event.error_message = str(exc)
        return

    # Execute the concrete pipeline.
    try:
        handler(preprocessing_event, sql_client)
        preprocessing_event.status = "succeeded"
        preprocessing_event.error_message = None
    except Exception as exc: 
        logger.error(
            "Error while running preprocessing pipeline %r for event_id=%s: %s",
            getattr(preprocessing_event, "pipeline_name", None),
            getattr(preprocessing_event, "id", None),
            exc,
        )
        preprocessing_event.status = "failed"
        preprocessing_event.error_message = str(exc)
        # We arent going to re-raise the exception here, the orchestration layer will inspect the status.
