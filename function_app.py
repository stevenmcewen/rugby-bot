import azure.functions as func
import json

from functions.config.settings import get_settings
from functions.sql.sql_client import SqlClient
from functions.data_ingestion.integration_services import (
    ingest_historical_kaggle_results,
    ingest_rugby365_fixtures,
    ingest_rugby365_results,
)
from functions.data_preprocessing.services import build_feature_tables
from functions.logging.logger import get_logger
from functions.ml_models.services import score_upcoming_matches, train_models
from functions.notifications.services import (
    generate_weekend_predictions,
    send_prediction_email,
)

app = func.FunctionApp()

logger = get_logger(__name__)
settings = get_settings()
sql_client = SqlClient(settings)


@app.route(route="IngestHistoricalKaggleResults", auth_level=func.AuthLevel.ANONYMOUS)
def IngestHistoricalKaggleResults(req: func.HttpRequest) -> func.HttpResponse:
    """
    Ingest historical Kaggle results data into the database.
    """
    logger.info("IngestHistoricalKaggleResults triggered.")

    # Start a system event for the ingestion.
    system_event = sql_client.start_system_event(
        function_name="IngestHistoricalKaggleResults",
        trigger_type="http",
        event_type="ingestion",
    )
    try:
        # Ingest the historical results.
        ingest_historical_kaggle_results(
            sql_client=sql_client,
            system_event_id=system_event.id,
        )
        # Best-effort completion of the system event.
        try:
            sql_client.complete_system_event(
                system_event_id=system_event.id,
                status="succeeded",
            )
        except Exception as ce:  # pragma: no cover - defensive logging
            logger.exception("Failed to mark system_event as succeeded: %s", ce)

        # Return a success response.
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "ok",
                    "message": "Historical Kaggle results ingestion triggered",
                    "system_event_id": str(system_event.id),
                }
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        # Log the exception.
        logger.exception("IngestHistoricalKaggleResults failed.")

        # Best-effort completion of the failed system event.
        try:
            sql_client.complete_system_event(
                system_event_id=system_event.id,
                status="failed",
                details=str(exc),
            )
        except Exception as ce:  # pragma: no cover - defensive logging
            logger.exception("Failed to mark system_event as failed: %s", ce)

        # Return a failed response.
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": "Historical Kaggle results ingestion failed",
                    "system_event_id": str(system_event.id),
                }
            ),
            status_code=500,
            mimetype="application/json",
        )

# Ingest Rugby365 results data into the database.
@app.schedule(
    schedule="0 0 1 * * *", # 01:00 UTC daily
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def IngestRugby365ResultsFunction(timer: func.TimerRequest) -> None:
    """
    Ingest Rugby365 results data into the database.
    """

    logger.info("IngestRugby365ResultsFunction triggered.")

    system_event = sql_client.start_system_event(
        function_name="IngestRugby365ResultsFunction",
        trigger_type="timer",
        event_type="ingestion",
    )

    try:
        ingest_rugby365_results(sql_client=sql_client, system_event_id=system_event.id)
        sql_client.complete_system_event(
            system_event_id=system_event.id,
            status="succeeded",
        )
    except Exception as exc: 
        logger.exception("IngestRugby365ResultsFunction failed.")
        sql_client.complete_system_event(
            system_event_id=system_event.id,
            status="failed",
            details=str(exc),
        )

# Ingest Rugby365 fixtures data into the database.
@app.schedule(
    schedule="0 30 1 * * *",  # 01:30 UTC daily
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def IngestRugby365FixturesFunction(timer: func.TimerRequest) -> None:
    """
    Ingest Rugby365 fixtures data into the database.
    """

    logger.info("IngestRugby365FixturesFunction triggered.")

    system_event = sql_client.start_system_event(
        function_name="IngestRugby365FixturesFunction",
        trigger_type="timer",
        event_type="ingestion",
    )

    try:
        ingest_rugby365_fixtures(sql_client=sql_client, system_event_id=system_event.id)
        sql_client.complete_system_event(
            system_event_id=system_event.id,
            status="succeeded",
        )
    except Exception as exc: 
        logger.exception("IngestRugby365FixturesFunction failed.")
        sql_client.complete_system_event(
            system_event_id=system_event.id,
            status="failed",
            details=str(exc),
        )
        raise

# # Build feature tables from the ingested data.
# @app.schedule(
#     schedule="0 0 3 * * *",  # 03:00 UTC daily, after ingestion
#     arg_name="timer",
#     run_on_startup=False,
#     use_monitor=True,
# )
# def BuildFeatureTablesFunction(timer: func.TimerRequest) -> None:
#     """
#     Preprocessing (Silver):
#     Read raw data from Blob (bronze), transform into clean tabular schemas,
#     and write model-ready tables to Azure SQL (silver) using managed identity.
#     """
#     if not settings.enable_scheduled_functions:
#         logger.info("BuildFeatureTablesFunction skipped (scheduled functions disabled).")
#         return

#     logger.info("BuildFeatureTablesFunction triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="BuildFeatureTablesFunction",
#         trigger_type="timer",
#         event_type="preprocessing",
#     )

#     try:
#         build_feature_tables()
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#     except Exception as exc:  # pragma: no cover - defensive logging
#         logger.exception("BuildFeatureTablesFunction failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         raise


# # Train models on the feature tables.
# @app.schedule(
#     schedule="0 0 4 * * MON",  # 04:00 UTC every Monday
#     arg_name="timer",
#     run_on_startup=False,
#     use_monitor=True,
# )
# def TrainModelsFunction(timer: func.TimerRequest) -> None:
#     """
#     ML Models - Training:
#     Periodically retrain prediction models on silver data (e.g. rolling window
#     of recent seasons) and persist model artifacts + metadata.
#     """
#     if not settings.enable_scheduled_functions:
#         logger.info("TrainModelsFunction skipped (scheduled functions disabled).")
#         return

#     logger.info("TrainModelsFunction triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="TrainModelsFunction",
#         trigger_type="timer",
#         event_type="training",
#     )

#     try:
#         train_models()
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#     except Exception as exc:  # pragma: no cover - defensive logging
#         logger.exception("TrainModelsFunction failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         raise


# # Score upcoming matches using the trained models.
# @app.schedule(
#     schedule="0 0 6 * * *",  # 06:00 UTC daily
#     arg_name="timer",
#     run_on_startup=False,
#     use_monitor=True,
# )
# def ScoreUpcomingMatchesFunction(timer: func.TimerRequest) -> None:
#     """
#     ML Models - Scoring:
#     Generate predictions for upcoming fixtures using the latest trained models
#     and store predictions for downstream consumption.
#     """
#     if not settings.enable_scheduled_functions:
#         logger.info("ScoreUpcomingMatchesFunction skipped (scheduled functions disabled).")
#         return

#     logger.info("ScoreUpcomingMatchesFunction triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="ScoreUpcomingMatchesFunction",
#         trigger_type="timer",
#         event_type="scoring",
#     )

#     try:
#         score_upcoming_matches()
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#     except Exception as exc:  # pragma: no cover - defensive logging
#         logger.exception("ScoreUpcomingMatchesFunction failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         raise


# # Generate weekend predictions using the trained models.
# @app.schedule(
#     schedule="0 0 9 * * FRI",  # 09:00 UTC every Friday (weekend preview)
#     arg_name="timer",
#     run_on_startup=False,
#     use_monitor=True,
# )
# def GenerateWeekendPredictionsFunction(timer: func.TimerRequest) -> None:
#     """
#     Notifications - Preparation:
#     Select upcoming weekend fixtures, fetch model predictions, and enqueue
#     a summary payload for email delivery.
#     """
#     if not settings.enable_scheduled_functions:
#         logger.info("GenerateWeekendPredictionsFunction skipped (scheduled functions disabled).")
#         return

#     logger.info("GenerateWeekendPredictionsFunction triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="GenerateWeekendPredictionsFunction",
#         trigger_type="timer",
#         event_type="notifications_prepare",
#     )

#     try:
#         generate_weekend_predictions()
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#     except Exception as exc:  # pragma: no cover - defensive logging
#         logger.exception("GenerateWeekendPredictionsFunction failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         raise


# # Send prediction email using the generated weekend predictions.
# @app.queue_trigger(
#     arg_name="msg",
#     queue_name="prediction-email-queue",
#     connection="AzureWebJobsStorage",
# )
# def SendPredictionEmailFunction(msg: func.QueueMessage) -> None:
#     """
#     Notifications - Delivery:
#     Consume prepared prediction payloads from a queue and send email summaries
#     of fixtures and model predictions.
#     """
#     body = msg.get_body().decode("utf-8")

#     system_event = sql_client.start_system_event(
#         function_name="SendPredictionEmailFunction",
#         trigger_type="queue",
#         event_type="notifications_delivery",
#     )

#     try:
#         logger.info(
#             "SendPredictionEmailFunction triggered with message body length=%d", len(body)
#         )
#         send_prediction_email(body)
#         # Best-effort completion of the system event.
#         try:
#             sql_client.complete_system_event(
#                 system_event_id=system_event.id,
#                 status="succeeded",
#             )
#         except Exception as ce:  # pragma: no cover - defensive logging
#             logger.exception("Failed to mark system_event as succeeded: %s", ce)
#     except Exception as exc:  # pragma: no cover - defensive logging
#         logger.exception("SendPredictionEmailFunction failed.")
#         # Best-effort completion of the failed system event.
#         try:
#             sql_client.complete_system_event(
#                 system_event_id=system_event.id,
#                 status="failed",
#                 details=str(exc),
#             )
#         except Exception as ce:  # pragma: no cover - defensive logging
#             logger.exception("Failed to mark system_event as failed: %s", ce)
#         raise