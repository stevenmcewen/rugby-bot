from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class AppSettings:
    """
    Strongly-typed application configuration.

    Values are primarily sourced from environment variables, which in Azure
    Functions come from Application Settings or `local.settings.json` in
    local development.
    """

    # Environment / deployment
    environment: Optional[str] = None

    # Azure SQL (silver layer)
    sql_server: Optional[str] = None
    sql_database: Optional[str] = None

    # Storage
    storage_connection: Optional[str] = None
    raw_container_name: str = "raw"

    # Email / notifications
    email_from: Optional[str] = None
    email_to: Optional[str] = None
    email_subject_prefix: str = "[rugby-bot]"

    # ML configuration
    model_name: str = "default_rugby_model"
    training_window_seasons: int = 5

    # Feature flags / switches
    enable_scheduled_functions: bool = False


@lru_cache()
def get_settings() -> AppSettings:
    """
    Load and cache application settings.

    This is called once per worker process and reused across function
    invocations, so configuration access is cheap and consistent.
    """

    return AppSettings(
        environment=os.getenv("ENVIRONMENT"),
        sql_server=os.getenv("SQL_SERVER"),
        sql_database=os.getenv("SQL_DATABASE"),
        storage_connection=os.getenv("AzureWebJobsStorage"),
        email_from=os.getenv("EMAIL_FROM"),
        email_to=os.getenv("EMAIL_TO"),
        email_subject_prefix=os.getenv("EMAIL_SUBJECT_PREFIX", "[rugby-bot]"),
        model_name=os.getenv("MODEL_NAME", "default_rugby_model"),
        training_window_seasons=int(os.getenv("TRAINING_WINDOW_SEASONS", "5")),
        enable_scheduled_functions=os.getenv("ENABLE_SCHEDULED_FUNCTIONS", "false").lower() == "true",
    )


