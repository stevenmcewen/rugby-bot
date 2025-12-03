from __future__ import annotations

from functions.config.settings import get_settings
from functions.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def build_feature_tables() -> None:
    """
    Orchestrate preprocessing from bronze (Blob) to silver (Azure SQL).

    This is where feature engineering, cleansing, and loading into SQL
    will be implemented.
    """

    logger.info(
        "Building feature tables in silver layer (sql_db=%s)",
        settings.sql_database,
    )
    # TODO: implement preprocessing and SQL writes.


