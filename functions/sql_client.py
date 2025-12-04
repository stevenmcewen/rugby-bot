from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID, uuid4

from functions.config.settings import AppSettings
from functions.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemEvent:
    """
    Lightweight representation of a row in dbo.system_events.

    This object is returned by the SQL client when a new system event
    is started, so callers can propagate the event id into downstream
    logic (e.g. ingestion_events).
    """

    id: UUID


class SqlClient:
    """
    Thin wrapper around Azure SQL access for system / ingestion metadata.

    Responsibilities:
    - Manage connections to Azure SQL (using managed identity in production).
    - Provide high-level methods for:
      - Starting / completing rows in dbo.system_events.
      - (Future) Writing rows to dbo.ingestion_events.

    NOTE: This is currently a shell; the actual connection and SQL execution
    need to be implemented using your preferred driver (e.g. pyodbc or
    sqlalchemy) and the managed identity pattern described in the README.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

        logger.info(
            "SqlClient initialised for server=%s, database=%s",
            settings.sql_server,
            settings.sql_database,
        )

        # TODO: Implement real connection pooling / engine initialisation here.
        # For example, using pyodbc + Azure AD access token, or SQLAlchemy.

    def start_system_event(
        self,
        *,
        function_name: str,
        trigger_type: str,
        event_type: str,
        status: str = "started",
        details: Optional[str] = None,
    ) -> SystemEvent:
        """
        Insert a new row into dbo.system_events and return its id.

        Intended usage pattern:
        - Call at the very start of an Azure Function.
        - Capture the returned id and pass it into downstream services
          (e.g. ingestion) so they can link ingestion_events to it.
        """

        event_id = uuid4()

        # TODO: Replace this logging stub with a real INSERT into dbo.system_events, e.g.:
        # INSERT INTO dbo.system_events (id, function_name, trigger_type, event_type, status, details)
        # VALUES (@id, @function_name, @trigger_type, @event_type, @status, @details);
        logger.info(
            "Starting system_event id=%s function=%s trigger=%s type=%s status=%s",
            event_id,
            function_name,
            trigger_type,
            event_type,
            status,
        )
        if details:
            logger.info("system_event id=%s details=%s", event_id, details)

        return SystemEvent(id=event_id)

    def complete_system_event(
        self,
        *,
        system_event_id: UUID,
        status: str,
        details: Optional[str] = None,
    ) -> None:
        """
        Mark an existing dbo.system_events row as completed (success / failure).

        Intended usage pattern:
        - Call in a finally/except block at the end of an Azure Function.
        """

        # TODO: Replace this logging stub with a real UPDATE to dbo.system_events, e.g.:
        # UPDATE dbo.system_events
        # SET status = @status,
        #     completed_at = SYSUTCDATETIME(),
        #     details = @details
        # WHERE id = @system_event_id;
        logger.info(
            "Completing system_event id=%s with status=%s",
            system_event_id,
            status,
        )
        if details:
            logger.info("system_event id=%s completion details=%s", system_event_id, details)

        # No return value; errors should be surfaced via exceptions from the underlying driver.


