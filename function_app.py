import azure.functions as func
import json

from functions.config.settings import get_settings
from functions.data_ingestion.services import (
    ingest_historical_results,
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


@app.route(route="IngestHistoricalResults", auth_level=func.AuthLevel.ANONYMOUS)
def IngestHistoricalResults(req: func.HttpRequest) -> func.HttpResponse:
    """
    Ingest historical results data into the database.
    """
    logger.info("IngestHistoricalResults triggered.")
    ingest_historical_results()
    return func.HttpResponse(
        json.dumps(
            {
                "status": "ok",
                "message": "Historical results ingestion triggered",
            }
        ),
        status_code=200,
        mimetype="application/json",
    )

# Ingest Rugby365 results data into the database.
@app.schedule(
    schedule="0 0 1 * * *",  # 01:00 UTC daily for Rugby365 results
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def IngestRugby365ResultsFunction(timer: func.TimerRequest) -> None:
    """
    Ingest Rugby365 results data into the database.
    """
    if not settings.enable_scheduled_functions:
        logger.info("IngestRugby365ResultsFunction skipped (scheduled functions disabled).")
        return
    logger.info("IngestRugby365ResultsFunction triggered.")
    ingest_rugby365_results()

# Ingest Rugby365 fixtures data into the database.
@app.schedule(
    schedule="0 0 1 * * *",  # 01:00 UTC daily
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def IngestRugby365FixturesFunction(timer: func.TimerRequest) -> None:
    """
    Ingest Rugby365 fixtures data into the database.
    """
    if not settings.enable_scheduled_functions:
        logger.info("IngestRugby365FixturesFunction skipped (scheduled functions disabled).")
        return
    logger.info("IngestRugby365FixturesFunction triggered.")
    ingest_rugby365_fixtures()


# Build feature tables from the ingested data.
@app.schedule(
    schedule="0 0 3 * * *",  # 03:00 UTC daily, after ingestion
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def BuildFeatureTablesFunction(timer: func.TimerRequest) -> None:
    """
    Preprocessing (Silver):
    Read raw data from Blob (bronze), transform into clean tabular schemas,
    and write model-ready tables to Azure SQL (silver) using managed identity.
    """
    if not settings.enable_scheduled_functions:
        logger.info("BuildFeatureTablesFunction skipped (scheduled functions disabled).")
        return
    logger.info("BuildFeatureTablesFunction triggered.")
    build_feature_tables()


# Train models on the feature tables.
@app.schedule(
    schedule="0 0 4 * * MON",  # 04:00 UTC every Monday
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def TrainModelsFunction(timer: func.TimerRequest) -> None:
    """
    ML Models - Training:
    Periodically retrain prediction models on silver data (e.g. rolling window
    of recent seasons) and persist model artifacts + metadata.
    """
    if not settings.enable_scheduled_functions:
        logger.info("TrainModelsFunction skipped (scheduled functions disabled).")
        return
    logger.info("TrainModelsFunction triggered.")
    train_models()


# Score upcoming matches using the trained models.
@app.schedule(
    schedule="0 0 6 * * *",  # 06:00 UTC daily
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def ScoreUpcomingMatchesFunction(timer: func.TimerRequest) -> None:
    """
    ML Models - Scoring:
    Generate predictions for upcoming fixtures using the latest trained models
    and store predictions for downstream consumption.
    """
    if not settings.enable_scheduled_functions:
        logger.info("ScoreUpcomingMatchesFunction skipped (scheduled functions disabled).")
        return
    logger.info("ScoreUpcomingMatchesFunction triggered.")
    score_upcoming_matches()


# Generate weekend predictions using the trained models.
@app.schedule(
    schedule="0 0 9 * * FRI",  # 09:00 UTC every Friday (weekend preview)
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def GenerateWeekendPredictionsFunction(timer: func.TimerRequest) -> None:
    """
    Notifications - Preparation:
    Select upcoming weekend fixtures, fetch model predictions, and enqueue
    a summary payload for email delivery.
    """
    if not settings.enable_scheduled_functions:
        logger.info("GenerateWeekendPredictionsFunction skipped (scheduled functions disabled).")
        return
    logger.info("GenerateWeekendPredictionsFunction triggered.")
    generate_weekend_predictions()


# Send prediction email using the generated weekend predictions.
@app.queue_trigger(
    arg_name="msg",
    queue_name="prediction-email-queue",
    connection="AzureWebJobsStorage",
)
def SendPredictionEmailFunction(msg: func.QueueMessage) -> None:
    """
    Notifications - Delivery:
    Consume prepared prediction payloads from a queue and send email summaries
    of fixtures and model predictions.
    """
    body = msg.get_body().decode("utf-8")
    logger.info(
        "SendPredictionEmailFunction triggered with message body length=%d", len(body)
    )
    send_prediction_email(body)