from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID
import time
import sqlalchemy as sa
from azure.identity import DefaultAzureCredential
import pyodbc
import struct 
import pandas as pd
import json

from functions.config.settings import AppSettings
from functions.logging.logger import get_logger
from functions.utils.utils import normalize_dataframe_dtypes

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

        logger.info("SQL config - server=%s, database=%s", settings.sql_server, settings.sql_database)

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
            avoid token‑expiry issues with long‑lived pools. It also
            adds some basic retry logic to improve resilience during
            cold starts (e.g. when SQL or managed identity is still
            waking up).
            """
            max_attempts = 3
            delay_seconds = 5
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return pyodbc.connect(
                        odbc_conn_str,
                        attrs_before=self.get_token(),
                    )
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "SQL connection attempt %s/%s failed: %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay_seconds)
                        # exponential backoff pattern
                        delay_seconds *= 2

            logger.error("SQL connection failed after %s attempts", max_attempts)
            raise last_exc

        # fast_executemany is set to False to avoid the issue of the SQL Server automatically setting incorrect string max lengths,
        # resulting in buffer overflow errors.
        self.engine = sa.create_engine(
            "mssql+pyodbc://",
            creator=_get_connection,
            pool_pre_ping=True,
            fast_executemany=False,
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
            
    def get_last_ingestion_event_created_at(
        self,
        integration_provider: str,
        integration_type: str,
    ) -> datetime | None:
        """
        Get the most recent ingestion_event.created_at for a given provider and type.

        Returns a timezone‑naive UTC datetime (as stored in SQL) or None if no
        successful/preprocessed ingestions exist.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT MAX(created_at)
                        FROM dbo.ingestion_events
                        WHERE integration_provider = :integration_provider
                          AND integration_type = :integration_type
                          AND status IN ('ingested', 'preprocessed', 'preprocessing');
                        """
                    ),
                    {
                        "integration_provider": integration_provider,
                        "integration_type": integration_type,
                    },
                )
                last_ingestion_event_created_at = result.scalar()

                if last_ingestion_event_created_at is None:
                    return None

                # SQLAlchemy may already give us a datetime; if it's a string, parse it.
                if isinstance(last_ingestion_event_created_at, datetime):
                    return last_ingestion_event_created_at

                # Fallback: attempt to parse common SQL datetime string formats.
                text_val = str(last_ingestion_event_created_at)
                try:
                    return datetime.fromisoformat(text_val)
                except ValueError:
                    # e.g. '2025-12-08 14:18:52.123456'
                    return datetime.strptime(text_val, "%Y-%m-%d %H:%M:%S.%f")
        except Exception as e:
            logger.error("Error getting last ingestion event created_at date: %s", e)
            raise

    ## ingestion source helpers ###
    def get_ingestion_events_by_status(
        self,
        status: str,
    ) -> list[dict]:
        """
        Get the ingestion_events with the given status and return them as a list of dictionaries.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT id, batch_id, system_event_id, integration_type, integration_provider, container_name, blob_path
                        FROM dbo.ingestion_events
                        WHERE status = :status;
                        """
                    ),
                    {
                        "status": status,
                    },
                )
                return result.fetchall()
        except Exception as e:
            logger.error("Error getting ingestion_events with status='%s': %s", status, e)
            raise

    ## source target mapping helpers ###
    def get_source_target_mapping(
        self,
        source_provider: str,
        source_type: str,
    ) -> list[dict]:
        """
        Get the source_target_mappings for a given source provider and type.
        Return the source_target_mapping as a list of dictionaries.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT source_provider, source_type, target_table, pipeline_name, execution_order, source_table
                        FROM dbo.preprocessing_source_target_mappings
                        WHERE source_provider = :source_provider AND source_type = :source_type
                        ORDER BY execution_order, pipeline_name;
                        """
                    ),
                    {
                        "source_provider": source_provider,
                        "source_type": source_type,
                    },
                )
                rows = result.fetchall()
                source_target_mappings = []
                for row in rows:
                    source_target_mapping = {
                        "source_provider": row[0],
                        "source_type": row[1],
                        "target_table": row[2],
                        "pipeline_name": row[3],
                        "execution_order": row[4],
                        "source_table": row[5],
                    }
                    source_target_mappings.append(source_target_mapping)
                return source_target_mappings
        except Exception as e:
            logger.error("Error getting source_target_mappings for source_provider='%s' and source_type='%s': %s", source_provider, source_type, e)
            raise

    
    ## preprocessing event helpers ###
    def create_preprocessing_event(
        self,
        *,
        preprocessing_plan: PreprocessingPlan,
    ) -> dict:
        """
        Create a new row in dbo.preprocessing_events and return it as a dictionary.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        INSERT INTO dbo.preprocessing_events (
                            batch_id,
                            system_event_id,
                            integration_type,
                            integration_provider,
                            container_name,
                            blob_path,
                            source_table,
                            target_table,
                            pipeline_name,
                            execution_order,
                            status,
                            error_message
                        )
                        OUTPUT INSERTED.id
                        VALUES (
                            :batch_id,
                            :system_event_id,
                            :integration_type,
                            :integration_provider,
                            :container_name,
                            :blob_path,
                            :source_table,
                            :target_table,
                            :pipeline_name,
                            :execution_order,
                            :status,
                            :error_message
                        );
                        """
                    ),
                    {
                        "batch_id": preprocessing_plan.batch_id,
                        "system_event_id": preprocessing_plan.system_event_id,
                        "integration_type": preprocessing_plan.integration_type,
                        "integration_provider": preprocessing_plan.integration_provider,
                        "container_name": preprocessing_plan.container_name,
                        "blob_path": preprocessing_plan.blob_path,
                        "source_table": preprocessing_plan.source_table,
                        "target_table": preprocessing_plan.target_table,
                        "pipeline_name": preprocessing_plan.pipeline_name,
                        "execution_order": preprocessing_plan.execution_order,
                        "status": "started",
                        "error_message": None,
                    },
                )
                row = result.first()
                conn.commit()
                preprocessing_event_id = UUID(str(row[0]))
                event_details = {
                    "id": preprocessing_event_id,
                    "batch_id": preprocessing_plan.batch_id,
                    "system_event_id": preprocessing_plan.system_event_id,
                    "integration_type": preprocessing_plan.integration_type,
                    "integration_provider": preprocessing_plan.integration_provider,
                    "container_name": preprocessing_plan.container_name,
                    "blob_path": preprocessing_plan.blob_path,
                    "source_table": preprocessing_plan.source_table,
                    "target_table": preprocessing_plan.target_table,
                    "pipeline_name": preprocessing_plan.pipeline_name,
                    "execution_order": preprocessing_plan.execution_order,
                    "status": "started",
                    "error_message": None,
                }
                logger.info(
                    "Created preprocessing event id=%s",
                    preprocessing_event_id,
                )
                return event_details
        except Exception as e:
            logger.error("Error creating preprocessing event: %s", e)
            raise

    def update_preprocessing_event(
        self,
        *,
        preprocessing_event_id: UUID,
        status: str,
        error_message: str|None = None,
    ) -> None:
        """
        Update an existing dbo.preprocessing_events row.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        UPDATE dbo.preprocessing_events
                        SET status = :status,
                            error_message = :error_message
                        WHERE id = :preprocessing_event_id;
                        """
                    ),
                    {
                        "status": status,
                        "error_message": error_message,
                        "preprocessing_event_id": preprocessing_event_id,
                    },
                )
                conn.commit()
                logger.info(
                    "Updated preprocessing event id=%s with status=%s error=%s",
                    preprocessing_event_id,
                    status,
                    error_message,
                )
        except Exception as e:
            logger.error("Error updating preprocessing event: %s", e)

    def get_preprocessing_events_by_batch_id(
        self,
        batch_id: UUID,
    ) -> list[PreprocessingEvent]:
        """
        Get the preprocessing_events for a given batch id.
        Return the preprocessing_events as a list of PreprocessingEvent objects.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT
                            id,
                            batch_id,
                            system_event_id,
                            integration_type,
                            integration_provider,
                            container_name,
                            blob_path,
                            source_table,
                            target_table,
                            pipeline_name,
                            execution_order,
                            status,
                            error_message
                        FROM dbo.preprocessing_events
                        WHERE batch_id = :batch_id;
                        """
                    ),
                    {
                        "batch_id": batch_id,
                    },
                )
                return result.fetchall()
        except Exception as e:
            logger.error("Error getting preprocessing events with batch_id='%s': %s", batch_id, e)
            raise

    ## schema helpers ###
    def get_schema(
        self,
        *,
        table_name: str | None = None,
        integration_type: str | None = None,
        integration_provider: str | None = None,
    ) -> list[dict]:
        """
        Get the logical schema (table + columns) from dbo.schema_tables
        and dbo.schema_columns.

        You can look up a schema in two main ways:

        - By physical table name: (This would be helpful for target tables)
            get_schema(table_name="dbo.rugby_results_clean")

        - By integration metadata: (This would be helpful for source tables)
            get_schema(
                integration_type="historical_results",
                integration_provider="kaggle",
            )

        Any combination of the filters provided will be AND-ed together.

        Returns:
            list[dict]: One dict per column with keys:
                - table_id
                - table_name
                - integration_type
                - integration_provider
                - description
                - column_id
                - column_name
                - data_type
                - is_required
                - ordinal_position
                - max_length
                - numeric_precision
                - numeric_scale
        """
        if not any([table_name, integration_type, integration_provider]):
            raise ValueError(
                "At least one of table_name, integration_type, or "
                "integration_provider must be provided to get_schema()."
            )

        where_clauses: list[str] = []
        params: dict[str, object] = {}
        ## build the where clause based on the filters provided
        if table_name is not None:
            if "." in table_name:
                schema, table_name = table_name.split(".", 1)
            where_clauses.append("st.table_name = :table_name")
            params["table_name"] = table_name

        if integration_type is not None:
            where_clauses.append("st.integration_type = :integration_type")
            params["integration_type"] = integration_type

        if integration_provider is not None:
            where_clauses.append("st.integration_provider = :integration_provider")
            params["integration_provider"] = integration_provider

        where_sql = ""
        # start the where clause with "WHERE" and join the other where clauses with "AND" if there are any
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT
                st.id                  AS table_id,
                st.table_name          AS table_name,
                st.integration_type    AS integration_type,
                st.integration_provider AS integration_provider,
                st.description         AS description,
                sc.id                  AS column_id,
                sc.column_name         AS column_name,
                sc.data_type           AS data_type,
                sc.is_required         AS is_required,
                sc.ordinal_position    AS ordinal_position,
                sc.max_length          AS max_length,
                sc.numeric_precision   AS numeric_precision,
                sc.numeric_scale       AS numeric_scale
            FROM dbo.schema_tables AS st
            JOIN dbo.schema_columns AS sc
              ON sc.table_id = st.id
            {where_sql}
            ORDER BY
                st.id,
                sc.ordinal_position,
                sc.column_name;
        """
        # execute the query and return the result
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(query), params)
                rows = result.fetchall()

                if not rows:
                    logger.warning(
                        "No schema rows found for filters: "
                        "table_name=%s, integration_type=%s, integration_provider=%s",
                        table_name,
                        integration_type,
                        integration_provider,
                    )
                    return []

                schema: list[dict] = []
                for row in rows:
                    schema.append(
                        {
                            "table_id": row.table_id,
                            "table_name": row.table_name,
                            "integration_type": row.integration_type,
                            "integration_provider": row.integration_provider,
                            "description": row.description,
                            "column_id": row.column_id,
                            "column_name": row.column_name,
                            "data_type": row.data_type,
                            "is_required": bool(row.is_required),
                            "ordinal_position": row.ordinal_position,
                            "max_length": row.max_length,
                            "numeric_precision": row.numeric_precision,
                            "numeric_scale": row.numeric_scale,
                        }
                    )
                return schema
        except Exception as e:
            logger.error(
                "Error getting schema for table_name='%s', integration_type='%s', integration_provider='%s': %s",
                table_name,
                integration_type,
                integration_provider,
                e,
            )
            raise

    ## generic data write helpers ###
    def write_dataframe_to_table(
        self,
        df: "pd.DataFrame",
        table_name: str,
        if_exists: str = "append",
    ) -> None:
        """
        Append the rows of a pandas DataFrame into a target SQL table.

        Args:
            df: The DataFrame to persist.
            table_name: The fully‑qualified table name, e.g. "dbo.matches".
            if_exists: Behaviour if the table already exists. Defaults to
                "append".
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("write_dataframe_to_table expected a pandas DataFrame")
        
        # tables will default to dbo schema if no schema is provided
        schema = "dbo"
        table = table_name

        # If someome provides a schema in the table name we will use it, otherwise we will use the default schema of dbo
        if "." in table_name:
            schema, table = table_name.split(".", 1)
        try:
            if df.empty:
                logger.info(
                    "write_dataframe_to_table: no rows to insert for table %s (schema=%s); skipping",
                    table,
                    schema,
                )
                return

            # get the number of rows before the insert
            rows_before = len(df)
            # We chunk rows for practical reasons (memory/latency). 
            chunksize = 1000

            # insert the data into the table
            df.to_sql(
                name=table,
                con=self.engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
            )
        except Exception as e:
            logger.error("Error writing DataFrame to table %s: %s", table_name, e)
            raise

    ## generic data read / maintenance helpers ###
    def read_table_to_dataframe(
        self,
        *,
        table_name: str,
        columns: list[str] | None = None,
        where_sql: str | None = None,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """
        Read a SQL table (optionally filtered) into a pandas DataFrame.

        Args:
            table_name: Fully-qualified table name (e.g. "dbo.InternationalMatchResults").
            columns: Optional list of column names to select. Defaults to "*".
            where_sql: Optional SQL snippet appended after WHERE (do not include 'WHERE').
            params: Optional parameters for where_sql (SQLAlchemy / pandas params).
        """
        if not table_name:
            raise ValueError("read_table_to_dataframe requires a non-empty table_name")

        # tables will default to dbo schema if no schema is provided
        schema = "dbo"
        table = table_name

        # If someome provides a schema in the table name we will use it, otherwise we will use the default schema of dbo
        if "." in table_name:
            schema, table = table_name.split(".", 1)
        # if columns are provided we will use them, otherwise we will use all columns
        select_cols = "*"
        if columns:
            select_cols = ", ".join(columns)
        # build the query
        query = f"SELECT {select_cols} FROM {schema}.{table}"
        # if where clause is provided we will add it to the query
        if where_sql:
            query += f" WHERE {where_sql}"
        # execute the query and return the result
        try:
            with self.engine.connect() as conn:
                # SQLAlchemy Connection.execute takes bound parameters as the *second positional*
                # argument (it does not accept a `params=` keyword argument).
                result = conn.execute(sa.text(query), params or {})
                rows = result.fetchall()
                return pd.DataFrame(rows, columns=result.keys())
        except Exception as e:
            logger.error("Error reading table %s into DataFrame: %s", table_name, e)
            raise

    def truncate_table(self, *, table_name: str) -> None:
        """
        Truncate a target table.
        """
        if not table_name:
            raise ValueError("truncate_table requires a non-empty table_name")
        # tables will default to dbo schema if no schema is provided
        schema = "dbo"
        table = table_name
        # If someome provides a schema in the table name we will use it, otherwise we will use the default schema of dbo
        if "." in table_name:
            schema, table = table_name.split(".", 1)
        # build the query
        query = f"TRUNCATE TABLE {schema}.{table}"
        # execute the query and return the result
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(query))
                conn.commit()
                logger.info("Truncated table %s", table_name)
        except Exception as e:
            logger.error("Error truncating table %s: %s", table_name, e)
            raise

    ## venue database helpers ###
    def get_venue_database(self) -> pd.DataFrame:
        """
        Load the rugby venue lookup table as a pandas DataFrame.

        Returns:
            pd.DataFrame with columns:
                - venue   (str): venue name as used in source data
                - city    (str): city where the venue is located
                - country (str): country where the venue is located

        Notes:
            - Only rows with IsActive = 1 are returned.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(
                    """
                    SELECT
                        VenueName AS venue,
                        City      AS city,
                        Country   AS country
                    FROM dbo.RugbyVenues
                    WHERE IsActive = 1;
                    """
                ))
                rows = result.fetchall()

            if not rows:
                logger.warning("get_venue_database: no rows found in dbo.RugbyVenues")
                # Return an empty DataFrame with the expected columns
                return pd.DataFrame(columns=["venue", "city", "country"])

            df = pd.DataFrame(rows, columns=result.keys())
            return df

        except Exception as e:
            logger.error("Error loading venue database from dbo.RugbyVenues: %s", e)
            raise

    ## model helpers ###
    def get_model_specs_by_model_group_key(self, model_group_key: str) -> dict:
        """
        Get the full model specification bundle for a given model_group_key.

        Returns a dict containing:
        - group: shared config (dataset sources, enabled flag)
        - models: list of models in the group (each with trainer + target)
        - columns: columns by role (entity/feature/target + optional weight)
        """
        try:
            with self.engine.connect() as conn:
                # 1) Model group
                group_result = conn.execute(
                    sa.text(
                        """
                        SELECT
                            ModelGroupKey,
                            TrainingDatasetSource,
                            ScoringDatasetSource,
                            IsEnabled,
                            ResultsTableName
                        FROM dbo.ModelGroup
                        WHERE ModelGroupKey = :model_group_key;
                        """
                    ),
                    {"model_group_key": model_group_key},
                )
                group_row = group_result.fetchone()
                if group_row is None:
                    raise ValueError(f"No ModelGroup found for model_group_key={model_group_key!r}")

                group_spec = {
                    "model_group_key": group_row[0],
                    "training_dataset_source": group_row[1],
                    "scoring_dataset_source": group_row[2],
                    "is_enabled": bool(group_row[3]),
                    "results_table_name": group_row[4],
                }

                if not group_spec["is_enabled"]:
                    raise ValueError(f"ModelGroup {model_group_key!r} is disabled")

                # 2) Models in group
                models_result = conn.execute(
                    sa.text(
                        """
                        SELECT
                            ModelKey,
                            TrainerKey,
                            TargetColumn,
                            PredictionType,
                            IsEnabled,
                            SampleWeightColumn,
                            TimeColumn,
                            ModelParametersJson,
                            TrainerParametersJson
                        FROM dbo.ModelRegistry
                        WHERE ModelGroupKey = :model_group_key
                        ORDER BY ModelKey;
                        """
                    ),
                    {"model_group_key": model_group_key},
                )
                model_rows = models_result.fetchall()
                if not model_rows:
                    raise ValueError(f"No models registered in ModelRegistry for model_group_key={model_group_key!r}")

                models = []
                for row in model_rows:
                    # First need to make sure that the ModelParametersJson is valid JSON and can be parsed
                    raw_model_params = row[7]
                    raw_trainer_params = row[8]

                    if raw_model_params is None or str(raw_model_params).strip() == "":
                        model_params = {}

                    else:
                        try:
                            model_params = json.loads(raw_model_params)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSON in ModelRegistry.ModelParametersJson for ModelKey={row[0]!r}: {e}"
                            ) from e
                        
                    if raw_trainer_params is None or str(raw_trainer_params).strip() == "":
                        trainer_params = {}
                    else:
                        try:
                            trainer_params = json.loads(raw_trainer_params)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSON in ModelRegistry.TrainerParametersJson for ModelKey={row[0]!r}: {e}"
                            ) from e
                    
                    # append the model spec
                    models.append(
                        {
                            "model_key": row[0],
                            "trainer_key": row[1],
                            "target_column": row[2],
                            "prediction_type": row[3],
                            "is_enabled": bool(row[4]),
                            "sample_weight_column": row[5],
                            "time_column": row[6],
                            "model_parameters": model_params,
                            "trainer_parameters": trainer_params,
                        }
                    )

                enabled_models = [m for m in models if m["is_enabled"]]
                if not enabled_models:
                    raise ValueError(f"All models in group {model_group_key!r} are disabled")

                # 3) Columns by role
                cols_result = conn.execute(
                    sa.text(
                        """
                        SELECT
                            ColumnName,
                            ColumnRole,
                            IsActive,
                            OrdinalPosition
                        FROM dbo.ModelColumn
                        WHERE ModelGroupKey = :model_group_key
                        ORDER BY
                            CASE ColumnRole
                                WHEN 'entity' THEN 1
                                WHEN 'feature' THEN 2
                                WHEN 'weight' THEN 3
                                WHEN 'target' THEN 4
                                ELSE 99
                            END,
                            OrdinalPosition;
                        """
                    ),
                    {"model_group_key": model_group_key},
                )
                col_rows = cols_result.fetchall()

                columns = {
                    "entity": [],
                    "feature": [],
                    "target": [],
                    "weight": None,   # single column name if present
                }

                for row in col_rows:
                    column_name = row[0]
                    column_role = row[1]
                    is_active = bool(row[2])

                    if not is_active:
                        continue

                    if column_role in ("entity", "feature", "target"):
                        columns[column_role].append(column_name)
                    elif column_role == "weight":
                        # allow at most one weight column
                        if columns["weight"] is not None and columns["weight"] != column_name:
                            raise ValueError(
                                f"Multiple active weight columns found for {model_group_key!r}: "
                                f"{columns['weight']!r} and {column_name!r}"
                            )
                        columns["weight"] = column_name

                if not columns["feature"]:
                    raise ValueError(f"No active feature columns found for model_group_key={model_group_key!r}")

                spec_bundle = {
                    **group_spec,
                    "models": enabled_models,
                    "columns": columns,
                }
                return spec_bundle

        except Exception as e:
            logger.error("Error getting model specs for model_group_key='%s': %s", model_group_key, e)
            raise

    def get_model_source_data(
        self,
        source_table: str,
        columns_to_select: list[str],
    ) -> pd.DataFrame:
        """
        Get the source data for a given model.
        Return the source data as a pandas DataFrame.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(f"SELECT {', '.join(columns_to_select)} FROM {source_table}"))
                rows = result.fetchall()
                df = pd.DataFrame(rows, columns=columns_to_select)
                normalized_df = normalize_dataframe_dtypes(df)
                return normalized_df
        except Exception as e:
            logger.error("Error getting model source data for source_table='%s' and columns_to_select='%s': %s", source_table, columns_to_select, e)
            raise


    def persist_artifact_metadata(
        self,
        *,
        system_event_id: str,
        model_key: str,
        trainer_key: str,
        prediction_type: str,
        target_column: str,
        schema_hash: str,
        artifact_version: int,
        blob_container: str,
        blob_path: str,
        metrics: dict | None,
    ) -> tuple[str, int]:
        """
        Insert a ModelArtifacts row and return (artifact_id, artifact_version).
        """
        try:
            metrics_json = json.dumps(metrics) if metrics is not None else None

            sql = sa.text("""
                INSERT INTO dbo.ModelArtifacts (
                    ArtifactId,
                    ArtifactVersion,
                    SystemEventId,
                    ModelKey,
                    TrainerKey,
                    PredictionType,
                    TargetColumn,
                    SchemaHash,
                    BlobContainer,
                    BlobPath,
                    Metrics
                )
                OUTPUT inserted.ArtifactId
                VALUES (
                    NEWID(),
                    :artifact_version,
                    :system_event_id,
                    :model_key,
                    :trainer_key,
                    :prediction_type,
                    :target_column,
                    :schema_hash,
                    :blob_container,
                    :blob_path,
                    :metrics
                );
            """)

            with self.engine.begin() as conn:
                artifact_id = conn.execute(
                    sql,
                    {
                        "artifact_version": artifact_version,
                        "system_event_id": system_event_id,
                        "model_key": model_key,
                        "trainer_key": trainer_key,
                        "prediction_type": prediction_type,
                        "target_column": target_column,
                        "schema_hash": schema_hash,
                        "blob_container": blob_container,
                        "blob_path": blob_path,
                        "metrics": metrics_json,
                    },
                ).scalar_one()

            return str(artifact_id), artifact_version

        except Exception as e:
            logger.error("Error persisting model artifact metadata: %s", e)
            raise

    def get_next_artifact_version(self, *, model_key: str) -> int:
        """
        Get the next artifact version number for a given model_key.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT ISNULL(MAX(ArtifactVersion), 0) + 1
                        FROM dbo.ModelArtifacts
                        WHERE ModelKey = :model_key;
                        """
                    ),
                    {"model_key": model_key},
                )
                next_version = result.scalar()
                return next_version
        except Exception as e:
            logger.error("Error getting next artifact version for model_key='%s': %s", model_key, e)
            raise

    def get_latest_model_artifact_details(self, *, model_key: str, trainer_key: str) -> dict | None:
        """
        Get the latest model artifact metadata for a given model_key.
        Returns a dict with the artifact metadata, or None if no artifact found.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT TOP 1
                            ArtifactId,
                            ArtifactVersion,
                            SystemEventId,
                            ModelKey,
                            TrainerKey,
                            PredictionType,
                            TargetColumn,
                            SchemaHash,
                            BlobContainer,
                            BlobPath,
                            Metrics,
                            CreatedAt
                        FROM dbo.ModelArtifacts
                        WHERE ModelKey = :model_key
                        AND TrainerKey = :trainer_key
                        ORDER BY ArtifactVersion DESC;
                        """
                    ),
                    {"model_key": model_key, "trainer_key": trainer_key},
                )
                row = result.fetchone()
                if row is None:
                    return None

                artifact_metadata = {
                    "artifact_id": str(row[0]),
                    "artifact_version": row[1],
                    "system_event_id": row[2],
                    "model_key": row[3],
                    "trainer_key": row[4],
                    "prediction_type": row[5],
                    "target_column": row[6],
                    "schema_hash": row[7],
                    "blob_container": row[8],
                    "blob_path": row[9],
                    "metrics": json.loads(row[10]) if row[10] else None,
                    "created_at": row[11],
                }
                return artifact_metadata
        except Exception as e:
            logger.error("Error getting latest artifact for model_key='%s': %s", model_key, e)
            raise
    

