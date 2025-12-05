from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
import sqlalchemy as sa
from azure.identity import DefaultAzureCredential
import pyodbc
import struct 

from functions.config.settings import AppSettings
from functions.logging.logger import get_logger

logger = get_logger(__name__)

# system event class ###
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
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.credential = DefaultAzureCredential()

        # Build a raw ODBC connection string. We use a custom creator
        # so that each new connection gets a *fresh* AAD token.
        odbc_conn_str = (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server=tcp:{settings.sql_server},1433;"
            f"Database={settings.sql_database};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        )

        def _get_connection():
            """
            Factory used by SQLAlchemy for each new DBAPI connection.

            This ensures we always attach a current access token and
            avoid token‑expiry issues with long‑lived pools.
            """
            return pyodbc.connect(
                odbc_conn_str,
                attrs_before=self.get_token(),
            )

        self.engine = sa.create_engine(
            "mssql+pyodbc://",
            creator=_get_connection,
            pool_pre_ping=True,
        )

        logger.info(
            "SqlClient connected: env=%s server=%s db=%s",
            settings.environment,
            settings.sql_server,
            settings.sql_database,
        )
    
    ## connection helpers ###
    def get_token(self) -> dict:
        """
        Get an Azure AD access token for the SQL Server.

        returns a dictionary that can be passed to the connect_args parameter of the SQL Alchemy engine.
        """
        token = self.credential.get_token("https://database.windows.net/.default").token
        # ODBC driver expects raw binary format
        token_bytes = token.encode("utf-16-le")
        exptoken = struct.pack("=i", len(token_bytes)) + token_bytes
        token = {1256: exptoken}
        return token
    
    ## event helpers ###
    def start_system_event(
        self,
        *,
        function_name: str,
        trigger_type: str,
        event_type: str,
        status: str = "started",
        details: str|None = None,
    ) -> SystemEvent:
        """
        Insert a new row into dbo.system_events and return its id.

        Intended usage pattern:
        - Call at the very start of an Azure Function.
        - Capture the returned id and pass it into downstream services
          (e.g. ingestion) so they can link ingestion_events to it.
        """
        try: 
            with self.engine.connect() as conn:
                result = conn.execute(sa.text("""
                    INSERT INTO dbo.system_events (function_name, trigger_type, event_type, status, details)
                    OUTPUT INSERTED.id
                    VALUES (:function_name, :trigger_type, :event_type, :status, :details);"""
                    ), 
                    {
                    "function_name": function_name,
                    "trigger_type": trigger_type,
                    "event_type": event_type,
                    "status": status,
                    "details": details,
                }
                )
                row = result.first()
                conn.commit()
                event_id = UUID(str(row[0]))
                logger.info(
                    "Starting system_event id=%s function=%s trigger=%s type=%s status=%s",
                    event_id,
                    function_name,
                    trigger_type,
                    event_type,
                    status,
                )
                return SystemEvent(id=event_id)
        except Exception as e:
            logger.error("Error starting system event: %s", e)
            raise


    def complete_system_event(
        self,
        *,
        system_event_id: UUID,
        status: str,
        details: str|None = None,
    ) -> None:
        """
        Mark an existing dbo.system_events row as completed (success / failure) at the end of a function run.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text("""
                    UPDATE dbo.system_events
                    SET status = :status,
                    completed_at = SYSUTCDATETIME(),
                    details = :details
                    WHERE id = :system_event_id;"""
                    ), 
                    {
                        "status": status,
                        "details": details,
                        "system_event_id": system_event_id,
                    }
                )
                conn.commit()
                logger.info(
                    "Completing system_event id=%s with status=%s",
                    system_event_id,
                    status,
                )
                if details:
                    logger.info("system_event id=%s completion details=%s", system_event_id, details)
        except Exception as e:
            logger.error("Error completing system event: %s", e)
            raise


    ## ingestion event helpers ###
    def start_ingestion_event(
        self,
        *,
        batch_id: UUID,
        system_event_id: UUID,
        container_name: str,
        integration_type: str,
        integration_provider: str,
        status: str,
        blob_path: str | None,
        error_message: str | None,
    ) -> UUID:
        """
        Insert a new row into dbo.ingestion_events and return its id.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        INSERT INTO dbo.ingestion_events (
                            batch_id,
                            system_event_id,
                            container_name,
                            integration_type,
                            integration_provider,
                            status,
                            blob_path,
                            error_message
                        )
                        OUTPUT INSERTED.id
                        VALUES (
                            :batch_id,
                            :system_event_id,
                            :container_name,
                            :integration_type,
                            :integration_provider,
                            :status,
                            :blob_path,
                            :error_message
                        );
                        """
                    ),
                    {
                        "batch_id": batch_id,
                        "system_event_id": system_event_id,
                        "container_name": container_name,
                        "integration_type": integration_type,
                        "integration_provider": integration_provider,
                        "status": status,
                        "blob_path": blob_path,
                        "error_message": error_message,
                    },
                )
                row = result.first()
                conn.commit()
                ingestion_event_id = UUID(str(row[0]))
                logger.info(
                    "Starting ingestion_event id=%s batch=%s system_event=%s container=%s type=%s provider=%s status=%s",
                    ingestion_event_id,
                    batch_id,
                    system_event_id,
                    container_name,
                    integration_type,
                    integration_provider,
                    status,
                )
                return ingestion_event_id
        except Exception as e:
            logger.error("Error starting ingestion event: %s", e)
            raise

    def update_ingestion_event(
        self,
        *,
        ingestion_event_id: UUID,
        status: str,
        blob_path: str|None = None,
        error_message: str|None = None,
    ) -> None:
        """
        Update an existing dbo.ingestion_events row.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        UPDATE dbo.ingestion_events
                        SET status = :status,
                            blob_path = :blob_path,
                            error_message = :error_message
                        WHERE id = :ingestion_event_id;
                        """
                    ),
                    {
                        "status": status,
                        "blob_path": blob_path,
                        "error_message": error_message,
                        "ingestion_event_id": ingestion_event_id,
                    },
                )
                conn.commit()

                logger.info(
                    "Updated ingestion_event id=%s with status=%s blob=%s error=%s",
                    ingestion_event_id,
                    status,
                    blob_path,
                    error_message,
                )
        except Exception as e:
            logger.error("Error updating ingestion event: %s", e)
            raise


