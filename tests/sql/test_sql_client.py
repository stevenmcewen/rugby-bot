from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

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
    def fake_create_engine(url: str, creator, pool_pre_ping: bool):
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


class DummyResult:
    def __init__(self, row_value: object | None = None):
        self._row_value = row_value

    def first(self):
        if self._row_value is None:
            return None
        return (self._row_value,)

    def scalar(self):
        return self._row_value


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


class DummyEngine:
    def __init__(self, connection: DummyConnection):
        self._connection = connection

    def connect(self):
        return self._connection


def _make_client_with_engine(connection: DummyConnection) -> sql_mod.SqlClient:
    client = sql_mod.SqlClient.__new__(sql_mod.SqlClient)
    client.engine = DummyEngine(connection)
    return client


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

    def fake_create_engine(url: str, creator, pool_pre_ping: bool):
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

    def fake_create_engine(url: str, creator, pool_pre_ping: bool):
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