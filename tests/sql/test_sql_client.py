from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pandas as pd

from functions.sql import sql_client as sql_mod


# Set up a dummy settings class for testing
@dataclass
class DummySettings:
    environment: str | None = "dev"
    sql_server: str | None = "server.database.windows.net"
    sql_database: str | None = "db"


# Does SqlClient init use the correct engine and credential?
# must use the correct engine and credential and set the correct environment, sql server and sql database
def test_sql_client_init_uses_engine_and_credential(monkeypatch):
    created = {}

    class FakeCredential:
        def get_token(self, scope: str):
            created["scope"] = scope
            return SimpleNamespace(token="dummy-token")

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def connect(self):
            raise AssertionError("connect should not be called in this test")

    # Does fake_create_engine use the correct url, creator and pool_pre_ping?
    # must use the correct url, creator and pool_pre_ping and set the correct url and pool_pre_ping
    def fake_create_engine(url: str, creator, pool_pre_ping: bool, **_kwargs):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping

        class FakeConnection:
            def close(self):
                pass

        # Monkeypatch pyodbc.connect used within creator so that it returns a fake
        # connection. Then call `creator()` once so that SqlClient.get_token()
        # is actually exercised, which lets us assert the requested scope.
        monkeypatch.setattr(
            sql_mod,
            "pyodbc",
            SimpleNamespace(connect=lambda *_a, **_k: FakeConnection()),
        )

        # This will call SqlClient.get_token() under the hood.
        creator()

        return FakeEngine()

    monkeypatch.setattr(sql_mod, "DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr(sql_mod.sa, "create_engine", fake_create_engine)

    settings = DummySettings()

    client = sql_mod.SqlClient(settings)

    assert isinstance(client.credential, FakeCredential)
    assert created["url"] == "mssql+pyodbc://"
    assert created["pool_pre_ping"] is True
    assert created["scope"] == "https://database.windows.net/.default"

# Does get_token build the correct odbc token?
# must build the correct odbc token and return the correct token dictionary
def test_get_token_builds_odbc_token():
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)  # bypass __init__

    class FakeCredential:
        def get_token(self, scope: str):
            assert scope == "https://database.windows.net/.default"
            return SimpleNamespace(token="abc123")

    client.credential = FakeCredential()

    token_dict = client.get_token()

    assert 1256 in token_dict
    value = token_dict[1256]
    assert isinstance(value, bytes)
    # Encoded token should end with the UTF‑16 bytes of "abc123"
    assert b"a\x00b\x00c\x001\x002\x003\x00" in value

# Create a dummy result for testing
class DummyResult:
    def __init__(self, row_value: object | None = None):
        self._row_value = row_value

    def first(self):
        if self._row_value is None:
            return None
        return (self._row_value,)

    def scalar(self):
        return self._row_value


# Create a dummy connection for testing
class DummyConnection:
    def __init__(self, row_value: str | None = None, should_fail: bool = False):
        self.row_value = row_value
        self.should_fail = should_fail
        self.executed = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        if self.should_fail:
            raise RuntimeError("db error")
        self.executed.append((args, kwargs))
        return DummyResult(self.row_value)

    def commit(self):
        self.committed = True


# Create a dummy engine for testing
class DummyEngine:
    def __init__(self, connection: DummyConnection):
        self._connection = connection

    def connect(self):
        return self._connection


def _make_client_with_engine(connection: DummyConnection) -> sql_mod.SqlClient:
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = DummyEngine(connection)
    return client

# Does the start_system_event function create a system event?
# must create a system event and return the correct system event id
def test_start_system_event_success(monkeypatch):
    event_id = uuid4()
    conn = DummyConnection(row_value=str(event_id))
    client = _make_client_with_engine(conn)

    # Ensure sa.text doesn't do anything surprising in tests.
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.start_system_event(
        function_name="fn",
        trigger_type="timer",
        event_type="ingestion",
        status="started",
        details="details",
    )

    assert isinstance(result, sql_mod.SystemEvent)
    assert result.id == event_id
    assert conn.committed is True
    assert len(conn.executed) == 1

# Does the start_system_event function raise an error when the connection fails?
# must raise an error when the connection fails
def test_start_system_event_error(monkeypatch):
    conn = DummyConnection(row_value=None, should_fail=True)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    with pytest.raises(RuntimeError):
        client.start_system_event(
            function_name="fn",
            trigger_type="timer",
            event_type="ingestion",
        )

# Does the complete_system_event function complete a system event?
# must complete a system event and return the correct system event id
def test_complete_system_event_success(monkeypatch):
    conn = DummyConnection()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    client.complete_system_event(
        system_event_id=uuid4(),
        status="completed",
        details="ok",
    )

    assert conn.committed is True
    assert len(conn.executed) == 1

# Does the complete_system_event function raise an error when the connection fails?
# must raise an error when the connection fails
def test_complete_system_event_error(monkeypatch):
    conn = DummyConnection(should_fail=True)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    with pytest.raises(RuntimeError):
        client.complete_system_event(
            system_event_id=uuid4(),
            status="failed",
            details="error",
        )

# Does the start_ingestion_event function create an ingestion event?
# must create an ingestion event and return the correct ingestion event id
def test_start_ingestion_event_success(monkeypatch):
    ingestion_id = uuid4()
    conn = DummyConnection(row_value=str(ingestion_id))
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.start_ingestion_event(
        batch_id=uuid4(),
        system_event_id=uuid4(),
        container_name="raw",
        integration_type="results",
        integration_provider="kaggle",
        status="started",
        blob_path=None,
        error_message=None,
    )

    assert result == ingestion_id
    assert conn.committed is True
    assert len(conn.executed) == 1

# Does the start_ingestion_event function raise an error when the connection fails?
# must raise an error when the connection fails
def test_start_ingestion_event_error(monkeypatch):
    conn = DummyConnection(row_value=None, should_fail=True)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    with pytest.raises(RuntimeError):
        client.start_ingestion_event(
            batch_id=uuid4(),
            system_event_id=uuid4(),
            container_name="raw",
            integration_type="results",
            integration_provider="kaggle",
            status="started",
            blob_path=None,
            error_message=None,
        )

# Does the update_ingestion_event function update an ingestion event?
# must update an ingestion event and return the correct ingestion event id
def test_update_ingestion_event_success(monkeypatch):
    conn = DummyConnection()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    client.update_ingestion_event(
        ingestion_event_id=uuid4(),
        status="ingested",
        blob_path="results/kaggle/dataset.csv",
        error_message=None,
    )

    assert conn.committed is True
    assert len(conn.executed) == 1

# Does the update_ingestion_event function raise an error when the connection fails?
# must raise an error when the connection fails
def test_update_ingestion_event_error(monkeypatch):
    conn = DummyConnection(should_fail=True)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    with pytest.raises(RuntimeError):
        client.update_ingestion_event(
            ingestion_event_id=uuid4(),
            status="failed",
            blob_path=None,
            error_message="err",
        )

# Does the get_last_ingestion_event_created_at function return None when the query result is NULL?
# must return None when the query result is NULL
def test_get_last_ingestion_event_created_at_returns_none(monkeypatch):
    """
    get_last_ingestion_event_created_at
    - must return None when the query result is NULL
    """
    conn = DummyConnection(row_value=None)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.get_last_ingestion_event_created_at(
        integration_provider="rugby365",
        integration_type="results",
    )

    assert result is None

# Does the get_last_ingestion_event_created_at function return a datetime unchanged when the DB already returns a datetime value?
# must return a datetime unchanged when the DB already returns a datetime value
def test_get_last_ingestion_event_created_at_returns_datetime_direct(monkeypatch):
    """
    get_last_ingestion_event_created_at
    - must return a datetime unchanged when the DB already returns a datetime value
    """
    dt = datetime(2025, 12, 8, 14, 30, 0)
    conn = DummyConnection(row_value=dt)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.get_last_ingestion_event_created_at(
        integration_provider="rugby365",
        integration_type="results",
    )

    assert isinstance(result, datetime)
    assert result == dt

# Does the get_last_ingestion_event_created_at function parse a common ISO datetime string value?
# must parse a common ISO datetime string value
def test_get_last_ingestion_event_created_at_parses_iso_string(monkeypatch):
    """
    get_last_ingestion_event_created_at
    - must parse common ISO datetime string values returned by SQL
    """
    value = "2025-12-08T14:18:52.123456"
    conn = DummyConnection(row_value=value)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.get_last_ingestion_event_created_at(
        integration_provider="rugby365",
        integration_type="results",
    )

    assert isinstance(result, datetime)
    assert result == datetime.fromisoformat(value)

# Does the SqlClient __init__ connection factory retry transient connection failures and eventually succeed?
# must retry transient connection failures and eventually succeed
def test_sql_client_connection_retries_then_succeeds(monkeypatch):
    """
    SqlClient __init__ connection factory
    - must retry transient connection failures and eventually succeed
    """
    attempts = {"count": 0}
    sleeps: list[int] = []
    created: dict = {}

    class FakeCredential:
        def get_token(self, scope: str):
            created["scope"] = scope
            return SimpleNamespace(token="dummy-token")

    def flaky_connect(*_args, **_kwargs):
        attempts["count"] += 1
        # Fail the first two attempts, succeed on the third.
        if attempts["count"] < 3:
            raise RuntimeError("temporary connect error")
        return SimpleNamespace(closed=False)

    def fake_create_engine(url: str, creator, pool_pre_ping: bool, **_kwargs):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping
        created["creator"] = creator
        # Engine itself is not exercised in this test, but must be constructible.
        return DummyEngine(DummyConnection())

    monkeypatch.setattr(sql_mod, "DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr(sql_mod.sa, "create_engine", fake_create_engine)
    monkeypatch.setattr(
        sql_mod,
        "pyodbc",
        SimpleNamespace(connect=flaky_connect),
    )
    monkeypatch.setattr(sql_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    settings = DummySettings()
    client = sql_mod.SqlClient(settings)

    # Exercise the captured creator to trigger the retry loop.
    creator = created["creator"]
    conn = creator()

    assert isinstance(client.credential, FakeCredential)
    assert created["url"] == "mssql+pyodbc://"
    assert created["pool_pre_ping"] is True
    assert created["scope"] == "https://database.windows.net/.default"
    assert attempts["count"] == 3
    assert isinstance(conn, SimpleNamespace)
    # Backoff should have slept between failed attempts: 5s then 10s.
    assert sleeps == [5, 10]

# Does the SqlClient __init__ connection factory give up and surface the last exception after max retries?
# must give up and surface the last exception after max retries
def test_sql_client_connection_retries_and_raises(monkeypatch):
    """
    SqlClient __init__ connection factory
    - must give up and surface the last exception after max retries
    """
    attempts = {"count": 0}
    sleeps: list[int] = []
    created: dict = {}

    class FakeCredential:
        def get_token(self, scope: str):
            created["scope"] = scope
            return SimpleNamespace(token="dummy-token")

    def always_failing_connect(*_args, **_kwargs):
        attempts["count"] += 1
        raise RuntimeError("permanent connect error")

    def fake_create_engine(url: str, creator, pool_pre_ping: bool, **_kwargs):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping
        created["creator"] = creator
        return DummyEngine(DummyConnection())

    monkeypatch.setattr(sql_mod, "DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr(sql_mod.sa, "create_engine", fake_create_engine)
    monkeypatch.setattr(
        sql_mod,
        "pyodbc",
        SimpleNamespace(connect=always_failing_connect),
    )
    monkeypatch.setattr(sql_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    settings = DummySettings()
    sql_mod.SqlClient(settings)

    creator = created["creator"]
    with pytest.raises(RuntimeError) as exc:
        creator()

    assert "permanent connect error" in str(exc.value)
    # Should have attempted the connection max_attempts times (3), with two sleeps.
    assert attempts["count"] == 3
    assert sleeps == [5, 10]

# Does the write_dataframe_to_table function use the dbo schema when no schema is provided?
# must use the dbo schema when no schema is provided
def test_write_dataframe_to_table_uses_dbo_when_schema_not_provided(monkeypatch):
    captured: dict = {}

    def fake_to_sql(self, *args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)

    df = pd.DataFrame({"a": [1], "b": [2]})
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = object()

    client.write_dataframe_to_table(df=df, table_name="InternationalMatchResults")

    assert captured["kwargs"]["schema"] == "dbo"
    assert captured["kwargs"]["name"] == "InternationalMatchResults"

# Does the write_dataframe_to_table function skip when the dataframe is empty?
# must skip when the dataframe is empty
def test_write_dataframe_to_table_empty_dataframe_skips(monkeypatch):
    called = {"to_sql": 0}

    def fake_to_sql(self, *args, **kwargs):
        called["to_sql"] += 1

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)

    df = pd.DataFrame(columns=["a", "b"])
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = object()

    client.write_dataframe_to_table(df=df, table_name="dbo.Target")
    assert called["to_sql"] == 0

# Does the write_dataframe_to_table function raise a TypeError when the dataframe is not a pandas DataFrame?
# must raise a TypeError when the dataframe is not a pandas DataFrame
def test_write_dataframe_to_table_non_dataframe_raises_type_error():
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = object()
    with pytest.raises(TypeError):
        client.write_dataframe_to_table(df="not-a-df", table_name="dbo.Target")  # type: ignore[arg-type]

# Does the get_ingestion_events_by_status function return the ingestion events by status?
# must return the ingestion events by status
def test_get_ingestion_events_by_status_returns_rows(monkeypatch):
    rows = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]
    conn = DummyConnection()
    client = _make_client_with_engine(conn)

    class DummyResultWithFetchall(DummyResult):
        def fetchall(self):
            return rows

    def fake_execute(*args, **kwargs):
        conn.executed.append((args, kwargs))
        return DummyResultWithFetchall()

    conn.execute = fake_execute  # type: ignore[method-assign]
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.get_ingestion_events_by_status(status="ingested")
    assert result == rows
    assert len(conn.executed) == 1

# Does the get_source_target_mapping function build the source target mapping?
# must build the source target mapping
def test_get_source_target_mapping_builds_dicts(monkeypatch):
    # emulate DB rows as tuples (as produced by SQLAlchemy Core)
    rows = [
        ("kaggle", "results", "dbo.T1", "p1", 1, None),
        ("kaggle", "results", "dbo.T2", "p2", 2, "dbo.T1"),
    ]
    conn = DummyConnection()
    client = _make_client_with_engine(conn)

    class DummyResultWithFetchall(DummyResult):
        def fetchall(self):
            return rows

    def fake_execute(*args, **kwargs):
        conn.executed.append((args, kwargs))
        return DummyResultWithFetchall()

    conn.execute = fake_execute  # type: ignore[method-assign]
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    mapping = client.get_source_target_mapping(source_provider="kaggle", source_type="results")
    assert mapping == [
        {
            "source_provider": "kaggle",
            "source_type": "results",
            "target_table": "dbo.T1",
            "pipeline_name": "p1",
            "execution_order": 1,
            "source_table": None,
        },
        {
            "source_provider": "kaggle",
            "source_type": "results",
            "target_table": "dbo.T2",
            "pipeline_name": "p2",
            "execution_order": 2,
            "source_table": "dbo.T1",
        },
    ]

# Does the create_preprocessing_event function create a preprocessing event?
# must create a preprocessing event
def test_create_preprocessing_event_success(monkeypatch):
    pre_id = uuid4()
    conn = DummyConnection(row_value=str(pre_id))
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    plan = SimpleNamespace(
        batch_id=uuid4(),
        system_event_id=uuid4(),
        integration_type="results",
        integration_provider="kaggle",
        container_name="raw",
        blob_path="a.csv",
        source_table=None,
        target_table="dbo.T",
        pipeline_name="p",
        execution_order=1,
    )

    event_details = client.create_preprocessing_event(preprocessing_plan=plan)
    assert event_details["id"] == pre_id
    assert event_details["batch_id"] == plan.batch_id
    assert event_details["system_event_id"] == plan.system_event_id
    assert event_details["integration_type"] == plan.integration_type
    assert event_details["integration_provider"] == plan.integration_provider
    assert event_details["container_name"] == plan.container_name
    assert event_details["blob_path"] == plan.blob_path
    assert event_details["source_table"] == plan.source_table
    assert event_details["target_table"] == plan.target_table
    assert event_details["pipeline_name"] == plan.pipeline_name
    assert event_details["execution_order"] == plan.execution_order
    assert event_details["status"] == "started"
    assert conn.committed is True


def test_read_table_to_dataframe_requires_table_name():
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = DummyEngine(DummyConnection())
    with pytest.raises(ValueError):
        client.read_table_to_dataframe(table_name="")


def test_read_table_to_dataframe_builds_query_and_returns_dataframe(monkeypatch):
    rows = [(1, 2), (3, 4)]

    class Result:
        def fetchall(self):
            return rows

        def keys(self):
            return ["a", "b"]

    class Conn(DummyConnection):
        def execute(self, *args, **kwargs):
            self.executed.append((args, kwargs))
            return Result()

    conn = Conn()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    df = client.read_table_to_dataframe(table_name="MyTable")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.to_dict(orient="records") == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    # Ensure the expected SQL was executed (default dbo schema).
    executed_sql = conn.executed[0][0][0]
    assert executed_sql == "SELECT * FROM dbo.MyTable"


def test_read_table_to_dataframe_supports_schema_columns_where_and_params(monkeypatch):
    rows = [(10, 20)]

    class Result:
        def fetchall(self):
            return rows

        def keys(self):
            return ["x", "y"]

    class Conn(DummyConnection):
        def execute(self, *args, **kwargs):
            self.executed.append((args, kwargs))
            return Result()

    conn = Conn()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    df = client.read_table_to_dataframe(
        table_name="silver.Facts",
        columns=["x", "y"],
        where_sql="x = :x",
        params={"x": 10},
    )

    assert df.to_dict(orient="records") == [{"x": 10, "y": 20}]
    executed_sql, kwargs = conn.executed[0][0][0], conn.executed[0][1]
    assert executed_sql == "SELECT x, y FROM silver.Facts WHERE x = :x"
    # SQLAlchemy Connection.execute receives bound params as the second positional arg.
    executed_args = conn.executed[0][0]
    assert executed_args[1] == {"x": 10}


def test_truncate_table_requires_table_name():
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = DummyEngine(DummyConnection())
    with pytest.raises(ValueError):
        client.truncate_table(table_name="")


def test_truncate_table_builds_query_and_commits(monkeypatch):
    conn = DummyConnection()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    client.truncate_table(table_name="MyTable")

    assert conn.committed is True
    executed_sql = conn.executed[0][0][0]
    assert executed_sql == "TRUNCATE TABLE dbo.MyTable"

# Does the update_preprocessing_event function update a preprocessing event?
# must update a preprocessing event
def test_update_preprocessing_event_success(monkeypatch):
    conn = DummyConnection()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    client.update_preprocessing_event(preprocessing_event_id=uuid4(), status="succeeded", error_message=None)
    assert conn.committed is True
    assert len(conn.executed) == 1

# Does the get_preprocessing_events_by_batch_id function return the preprocessing events by batch id?
# must return the preprocessing events by batch id
def test_get_preprocessing_events_by_batch_id_returns_rows(monkeypatch):
    rows = [SimpleNamespace(status="succeeded"), SimpleNamespace(status="failed")]
    conn = DummyConnection()
    client = _make_client_with_engine(conn)

    class DummyResultWithFetchall(DummyResult):
        def fetchall(self):
            return rows

    def fake_execute(*args, **kwargs):
        conn.executed.append((args, kwargs))
        return DummyResultWithFetchall()

    conn.execute = fake_execute  # type: ignore[method-assign]
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.get_preprocessing_events_by_batch_id(batch_id=uuid4())
    assert result == rows
    # Ensure the query includes new stage-2 columns so consumers can access them safely.
    executed_sql = conn.executed[0][0][0]
    assert "source_table" in executed_sql
    assert "execution_order" in executed_sql

# Does the get_schema function require at least one filter?
# must require at least one filter
def test_get_schema_requires_at_least_one_filter():
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    with pytest.raises(ValueError):
        client.get_schema()

# Does the get_schema function build the expected list?
# must build the expected list
def test_get_schema_builds_expected_list(monkeypatch):
    # Provide SQLAlchemy Row-like objects with attribute access.
    rows = [
        SimpleNamespace(
            table_id=uuid4(),
            table_name="dbo.Target",
            integration_type="results",
            integration_provider="kaggle",
            description="d",
            column_id=uuid4(),
            column_name="MatchDate",
            data_type="date",
            is_required=1,
            ordinal_position=1,
            max_length=None,
            numeric_precision=None,
            numeric_scale=None,
        )
    ]

    class Result:
        def fetchall(self):
            return rows

    class Conn(DummyConnection):
        def execute(self, *args, **kwargs):
            self.executed.append((args, kwargs))
            return Result()

    conn = Conn()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    schema = client.get_schema(table_name="dbo.Target")
    assert schema[0]["table_name"] == "dbo.Target"
    assert schema[0]["column_name"] == "MatchDate"
    assert schema[0]["is_required"] is True

# Does the get_venue_database function return the venue database?
# must return the venue database
def test_get_venue_database_returns_dataframe(monkeypatch):
    # Mimic SQLAlchemy result: rows + keys()
    rows = [("CAPE TOWN STADIUM", "Cape Town", "South Africa")]
    keys = lambda: ["venue", "city", "country"]

    class Result:
        def fetchall(self):
            return rows

        def keys(self):
            return keys()

    class Conn(DummyConnection):
        def execute(self, *args, **kwargs):
            self.executed.append((args, kwargs))
            return Result()

    conn = Conn()
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    df = client.get_venue_database()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["venue", "city", "country"]
    assert df.iloc[0]["city"] == "Cape Town"

# Does the get_last_ingestion_event_created_at function parse the SQL string format?
# must parse the SQL string format
def test_get_last_ingestion_event_created_at_parses_sql_string_format(monkeypatch):
    value = "2025-12-08 14:18:52.123456"
    conn = DummyConnection(row_value=value)
    client = _make_client_with_engine(conn)
    monkeypatch.setattr(sql_mod.sa, "text", lambda sql: sql)

    result = client.get_last_ingestion_event_created_at(
        integration_provider="rugby365",
        integration_type="results",
    )
    assert result == datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")