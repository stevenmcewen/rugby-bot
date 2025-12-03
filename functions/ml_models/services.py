from __future__ import annotations

from functions.config.settings import get_settings
from functions.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def train_models() -> None:
    """
    Train or retrain prediction models on silver-layer data.

    This function is expected to:
    - Load training data from Azure SQL.
    - Train models on a rolling window of recent seasons.
    - Persist model artifacts and metadata for later scoring.
    """

    logger.info(
        "Training models (model_name=%s, window_seasons=%d)",
        settings.model_name,
        settings.training_window_seasons,
    )
    # TODO: implement model training and persistence.


def score_upcoming_matches() -> None:
    """
    Score upcoming fixtures using the latest trained models.

    This function is expected to:
    - Read upcoming fixtures from SQL.
    - Load current model artifacts.
    - Write predictions back to SQL or another storage for consumption.
    """

    logger.info("Scoring upcoming matches using model %s", settings.model_name)
    # TODO: implement scoring pipeline.


