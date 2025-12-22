from __future__ import annotations

from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.data_sources.sql_client import SqlClient
from uuid import UUID


logger = get_logger(__name__)
settings = get_settings()

### Public entrypoint orchestrating model training ###
def orchestrate_model_training(
    *,
    sql_client: SqlClient,
    system_event_id: UUID,
    pipeline_name: str = "default_model_training",
    model_key: str,
) -> None:
    """
    High‑level orchestration entrypoint for the model training phase.
    Accepts:
    - sql_client: SqlClient instance
    - system_event_id: UUID of the system event
    - pipeline_name: Name of the pipeline to run
    - model_group_key: Key of the model group to train
    Returns:
    - None
    """

    logger.info("Starting model training orchestration with pipeline=%s", pipeline_name)

    pipeline = build_model_pipeline_for(pipeline_name)

    context: ModelContext = {
        "sql_client": sql_client,
        "system_event_id": system_event_id,
        "status": "started",
        "error_message": None,
        "model_group_key": model_group_key,
    }

    pipeline.run(context)

### Public entrypoint orchestrating model scoring ###
def orchestrate_model_scoring(
    *,
    sql_client: SqlClient,
    system_event_id: UUID,
    pipeline_name: str = "default_model_scoring",
    model_key: str,
) -> None:
    """
    High‑level orchestration entrypoint for the model scoring phase.
    Accepts:
    - sql_client: SqlClient instance
    - system_event_id: UUID of the system event
    - pipeline_name: Name of the pipeline to run
    - model_group_key: Key of the model group to score
    Returns:
    - None
    """

    logger.info("Starting model scoring orchestration with pipeline=%s", pipeline_name)

    pipeline = build_model_pipeline_for(pipeline_name)

    context: ModelContext = {
        "sql_client": sql_client,
        "system_event_id": system_event_id,
        "status": "started",
        "error_message": None,
        "model_group_key": model_group_key,
    }

    pipeline.run(context)

