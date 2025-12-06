from __future__ import annotations

from functions.config.settings import get_settings
from functions.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def generate_weekend_predictions() -> None:
    """
    Prepare a summary of weekend fixtures and model predictions.

    This will typically:
    - Query upcoming fixtures and associated predictions from SQL.
    - Construct a compact payload (e.g. dict/JSON) describing the games.
    - Enqueue the payload for email delivery.
    """

    logger.info("Generating weekend predictions summary for notifications.")
    # TODO: implement query + queue enqueue.


def send_prediction_email(payload: str) -> None:
    """
    Send an email containing fixtures and predictions.

    `payload` is expected to be a JSON string or similar structure produced
    by `generate_weekend_predictions`.
    """

    logger.info(
        "Sending prediction email from %s to %s",
        # settings.email_from,
        # settings.email_to,
    )
    # TODO: parse payload, render template, and send email via chosen provider.


