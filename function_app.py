import azure.functions as func
import json

from functions.config.settings import get_settings
from functions.sql.sql_client import SqlClient
from functions.data_ingestion.integration_services import (
    ingest_historical_kaggle_results,
    ingest_rugby365_fixtures,
    ingest_rugby365_results,
)
from functions.data_preprocessing.preprocessing_services import orchestrate_preprocessing
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

# Preprocess data into the database.
@app.schedule(
    schedule="0 0 3 * * *", # 03:00 UTC daily
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def PreprocessDataFunction(timer: func.TimerRequest) -> None:
    """
    Preprocess data into the database.
    """

    logger.info("PreprocessDataFunction triggered.")

    system_event = sql_client.start_system_event(
        function_name="PreprocessDataFunction",
        trigger_type="timer",
        event_type="preprocessing",
    )

    try:
        orchestrate_preprocessing(sql_client=sql_client, system_event_id=system_event.id)
        sql_client.complete_system_event(
            system_event_id=system_event.id,
            status="succeeded",
        )
    except Exception as exc: 
        logger.exception("PreprocessDataFunction failed.")
        sql_client.complete_system_event(
            system_event_id=system_event.id,
            status="failed",
            details=str(exc),
        )
        raise

# #------------------------------------------------------------------------------------------------
# # Testing functions
# #------------------------------------------------------------------------------------------------
# # insert test
# @app.route(route="IngestRugby365ResultsFunctionTest", auth_level=func.AuthLevel.ANONYMOUS)
# def IngestRugby365ResultsFunctionTest(req: func.HttpRequest) -> func.HttpResponse:
#     """
#     Ingest Rugby365 results data into the database.
#     """

#     logger.info("IngestRugby365ResultsFunction triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="IngestRugby365ResultsFunctionTest",
#         trigger_type="test",
#         event_type="ingestion",
#     )

#     try:
#         ingest_rugby365_results(sql_client=sql_client, system_event_id=system_event.id)
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#         return func.HttpResponse(
#             json.dumps(
#                 {
#                     "status": "ok",
#                     "message": "IngestRugby365ResultsFunctionTest triggered",
#                     "system_event_id": str(system_event.id),
#                 }
#             ),
#             status_code=200,
#             mimetype="application/json",
#         )
#     except Exception as exc: 
#         logger.exception("IngestRugby365ResultsFunctionTest failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         return func.HttpResponse(
#             json.dumps(
#                 {
#                     "status": "error",
#                     "message": "IngestRugby365ResultsFunctionTest failed",
#                     "system_event_id": str(system_event.id),
#                 }
#             ),
#             status_code=500,
#             mimetype="application/json",
#         )
#         raise

# # Ingest Rugby365 fixtures data into the database.
# @app.route(route="IngestRugby365FixturesFunctionTest", auth_level=func.AuthLevel.ANONYMOUS)
# def IngestRugby365FixturesFunctionTest(req: func.HttpRequest) -> func.HttpResponse:
#     """
#     Ingest Rugby365 fixtures data into the database.
#     """

#     logger.info("IngestRugby365FixturesFunctionTest triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="IngestRugby365FixturesFunctionTest",
#         trigger_type="test",
#         event_type="ingestion",
#     )

#     try:
#         ingest_rugby365_fixtures(sql_client=sql_client, system_event_id=system_event.id)
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#         return func.HttpResponse(
#             json.dumps(
#                 {
#                     "status": "ok",
#                     "message": "IngestRugby365FixturesFunctionTest triggered",
#                     "system_event_id": str(system_event.id),
#                 }
#             ),
#             status_code=200,
#             mimetype="application/json",
#         )
#     except Exception as exc: 
#         logger.exception("IngestRugby365FixturesFunctionTest failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         return func.HttpResponse(
#             json.dumps(
#                 {
#                     "status": "error",
#                     "message": "IngestRugby365FixturesFunctionTest failed",
#                     "system_event_id": str(system_event.id),
#                 }
#             ),
#             status_code=500,
#             mimetype="application/json",
#         )
#         raise

# # Preprocess test.
# @app.route(route="PreprocessTest", auth_level=func.AuthLevel.ANONYMOUS)
# def PreprocessTest(req: func.HttpRequest) -> func.HttpResponse:
#     """
#     Preprocess test.
#     """
#     logger.info("PreprocessTest triggered.")

#     system_event = sql_client.start_system_event(
#         function_name="PreprocessTest",
#         trigger_type="test",
#         event_type="preprocessing",
#     )

#     try:
#         orchestrate_preprocessing(
#             sql_client=sql_client, 
#             system_event_id=system_event.id
#         )
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="succeeded",
#         )
#         return func.HttpResponse(
#             json.dumps(
#                 {
#                     "status": "ok",
#                     "message": "PreprocessTest triggered",
#                     "system_event_id": str(system_event.id),
#                 }
#             ),
#             status_code=200,
#             mimetype="application/json",
#         )
#     except Exception as exc:
#         logger.exception("PreprocessTest failed.")
#         sql_client.complete_system_event(
#             system_event_id=system_event.id,
#             status="failed",
#             details=str(exc),
#         )
#         return func.HttpResponse(
#             json.dumps(
#                 {
#                     "status": "error",
#                     "message": "PreprocessTest failed",
#                     "system_event_id": str(system_event.id),
#                 }
#             ),
#             status_code=500,
#             mimetype="application/json",
#         )
#         raise
