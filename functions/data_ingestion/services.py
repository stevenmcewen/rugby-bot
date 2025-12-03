from __future__ import annotations

from functions.config.settings import get_settings
from functions.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def ingest_rugby_data() -> None:
    """
    Orchestrate ingestion of rugby data into the bronze layer.

    This function will eventually:
    - Download / read historical data (e.g. Kaggle bootstrap).
    - Scrape Rugby365 fixtures and results.
    - Write raw snapshots to Blob Storage using the configured container.
    """

    logger.info(
        "Ingesting rugby data into bronze layer (container=%s)",
        settings.raw_container_name,
    )
    # TODO: implement concrete ingestion steps.


