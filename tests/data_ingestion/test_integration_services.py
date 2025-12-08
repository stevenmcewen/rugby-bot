import types
from uuid import UUID, uuid4

import pytest

from functions.data_ingestion import integration_services as services


class DummyStep:
    def __init__(self, name: str):
        self.name = name
        self.called_with = []

    def __call__(self, context: services.IngestionContext) -> services.IngestionContext:
        self.called_with.append(context.copy())
        # Tag context so we can see ordering
        context[f"visited_{self.name}"] = True
        return context

# Does Integration Pipeline call steps in sequence, pass context between steps and call each step once?
def test_ingestion_pipeline_runs_steps_in_order():
    step1 = DummyStep("one")
    step2 = DummyStep("two")
    pipeline = services.IngestionPipeline(step1, step2)

    context = {"initial": True}

    result = pipeline.run(context)

    assert result["initial"] is True
    assert result["visited_one"] is True
    assert result["visited_two"] is True
    assert len(step1.called_with) == 1
    assert len(step2.called_with) == 1

# Does DownloadHistoricalDataStep call download_historical_results and pass the context to the next step?
def test_download_historical_data_step_success(monkeypatch):
    step = services.DownloadHistoricalDataStep()

    def fake_download(provider: str) -> tuple[str, str]:
        assert provider == "kaggle"
        return "/tmp/results.csv", "owner/dataset"

    monkeypatch.setattr(services, "download_historical_results", fake_download)

    ctx: services.IngestionContext = {"integration_provider": "kaggle"}

    result = step(ctx)

    assert result["local_integration_file_path"] == "/tmp/results.csv"
    assert result["integration_dataset"] == "owner/dataset"
    assert result["status"] == "started"

# Does DownloadHistoricalDataStep handle errors gracefully?
# must not crash the pipeline, must set the status to "failed" and set the error message
def test_download_historical_data_step_error(monkeypatch):
    step = services.DownloadHistoricalDataStep()

    def fake_download(provider: str) -> tuple[str, str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(services, "download_historical_results", fake_download)

    ctx: services.IngestionContext = {"integration_provider": "kaggle"}

    result = step(ctx)

    assert result["status"] == "failed"
    assert "boom" in result["error_message"]

# Does ScrapeResultsOrFixturesStep call scrape_values for results and pass the context to the next step?
# must call scrape_values with the correct provider/type and return the correct local file path and integration dataset
def test_scrape_results_or_fixtures_results(monkeypatch):
    step = services.ScrapeResultsOrFixturesStep()

    called = {}

    def fake_scrape_values(integration_provider: str, integration_type: str, sql_client) -> tuple[str, str]:
        called["provider"] = integration_provider
        called["type"] = integration_type
        called["sql_client"] = sql_client
        return "/tmp/results.csv", "results_dataset"

    monkeypatch.setattr(services, "scrape_values", fake_scrape_values)

    class FakeSqlClient:
        pass

    ctx: services.IngestionContext = {
        "integration_type": "results",
        "integration_provider": "rugby365",
        "sql_client": FakeSqlClient(),
    }

    result = step(ctx)

    assert called["provider"] == "rugby365"
    assert called["type"] == "results"
    assert result["local_integration_file_path"] == "/tmp/results.csv"
    assert result["integration_dataset"] == "results_dataset"
    assert result["status"] == "started"

# Does ScrapeResultsOrFixturesStep call scrape_values for fixtures and pass the context to the next step?
# must call scrape_values with the correct provider/type and return the correct local file path and integration dataset
def test_scrape_results_or_fixtures_fixtures(monkeypatch):
    step = services.ScrapeResultsOrFixturesStep()

    called = {}

    def fake_scrape_values(integration_provider: str, integration_type: str, sql_client) -> tuple[str, str]:
        called["provider"] = integration_provider
        called["type"] = integration_type
        called["sql_client"] = sql_client
        return "/tmp/fixtures.csv", "fixtures_dataset"

    monkeypatch.setattr(services, "scrape_values", fake_scrape_values)

    class FakeSqlClient:
        pass

    ctx: services.IngestionContext = {
        "integration_type": "fixtures",
        "integration_provider": "rugby365",
        "sql_client": FakeSqlClient(),
    }

    result = step(ctx)

    assert called["provider"] == "rugby365"
    assert called["type"] == "fixtures"
    assert result["local_integration_file_path"] == "/tmp/fixtures.csv"
    assert result["integration_dataset"] == "fixtures_dataset"
    assert result["status"] == "started"

# Does ScrapeResultsOrFixturesStep handle invalid integration type gracefully?
# must set the status to "failed", set the error message and propagate the error
def test_scrape_results_or_fixtures_invalid_type(monkeypatch):
    step = services.ScrapeResultsOrFixturesStep()

    class FakeSqlClient:
        pass

    def fake_scrape_values(integration_provider: str, integration_type: str, sql_client):
        raise ValueError(f"Unsupported integration type: {integration_type}")

    monkeypatch.setattr(services, "scrape_values", fake_scrape_values)

    ctx: services.IngestionContext = {
        "integration_type": "unknown",
        "integration_provider": "rugby365",
        "sql_client": FakeSqlClient(),
    }

    with pytest.raises(ValueError) as exc:
        step(ctx)

    assert ctx["status"] == "failed"
    assert "Unsupported integration type" in ctx["error_message"]

# Does WriteRawSnapshotsToBlobStep skip when status is failed?
# must not call BlobClient and set the status to "failed" and set the error message
def test_write_raw_snapshots_skips_when_failed(monkeypatch):
    # Ensure BlobClient is not even called when status is failed
    called = {"from_connection_string": False}

    monkeypatch.setattr(
        services,
        "BlobClient",
        types.SimpleNamespace(
            from_connection_string=lambda *_, **__: called.__setitem__(
                "from_connection_string", True
            ),
        ),
    )

    ctx: services.IngestionContext = {
        "status": "failed",
        "system_event_id": uuid4(),
        "ingestion_event_id": uuid4(),
    }

    result = services.WriteRawSnapshotsToBlobStep()(ctx)

    assert result is ctx
    assert called["from_connection_string"] is False

# Does WriteRawSnapshotsToBlobStep call BlobClient and pass the context to the next step?
# must call BlobClient with the correct connection string, container name and blob name and return the correct local file path and integration dataset
def test_write_raw_snapshots_success(monkeypatch):
    uploaded = {}

    class FakeBlobClient:
        def __init__(self, *_, **__):
            pass

        def upload_blob(self, data, overwrite: bool) -> None:
            # In production we pass an open file handle; in tests we just record it.
            uploaded["data"] = data
            uploaded["overwrite"] = overwrite

    def fake_from_connection_string(conn_str: str, container_name: str, blob_name: str):
        uploaded["conn_str"] = conn_str
        uploaded["container_name"] = container_name
        uploaded["blob_name"] = blob_name
        return FakeBlobClient()

    # Patch BlobClient factory
    monkeypatch.setattr(
        services, "BlobClient", types.SimpleNamespace(from_connection_string=fake_from_connection_string)
    )

    # Patch `open` used in the module so we don't need a real file on disk.
    class FakeFile:
        def __init__(self, path: str):
            self.path = path

        def read(self, *_a, **_k):
            return b"fake-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def fake_open(path: str, mode: str = "r", *_, **__):
        uploaded["opened_path"] = path
        uploaded["mode"] = mode
        return FakeFile(path)

    # The implementation uses the built-in open; patch it in this module's namespace.
    monkeypatch.setattr(services, "open", fake_open, raising=False)

    # Patch settings used inside the module
    services.settings = types.SimpleNamespace(
        storage_connection="UseDevelopmentStorage=true",
        raw_container_name="raw-container",
    )

    ctx: services.IngestionContext = {
        "status": "started",
        "raw_container_name": "raw-container",
        "integration_type": "results",
        "integration_provider": "kaggle",
        "integration_dataset": "owner/dataset",
        "local_integration_file_path": "/tmp/results.csv",
    }

    result = services.WriteRawSnapshotsToBlobStep()(ctx)

    expected_blob_name = "results/kaggle/owner/dataset.csv"

    assert uploaded["conn_str"] == "UseDevelopmentStorage=true"
    assert uploaded["container_name"] == "raw-container"
    assert uploaded["blob_name"] == expected_blob_name
    # Ensure we tried to open the correct local path in binary mode
    assert uploaded["opened_path"] == "/tmp/results.csv"
    assert uploaded["mode"] == "rb"
    assert uploaded["overwrite"] is True
    assert result["blob_snapshot_uri"] == expected_blob_name
    assert result["status"] == "ingested"

# Does WriteRawSnapshotsToBlobStep handle errors gracefully?
# must not crash the pipeline, must set the status to "failed" and set the error message
def test_write_raw_snapshots_error(monkeypatch):
    def fake_from_connection_string(*_, **__):
        class FakeBlobClient:
            def upload_blob(self, *_a, **_k):
                raise RuntimeError("upload failed")

        return FakeBlobClient()

    monkeypatch.setattr(
        services, "BlobClient", types.SimpleNamespace(from_connection_string=fake_from_connection_string)
    )
    services.settings = types.SimpleNamespace(
        storage_connection="conn",
        raw_container_name="raw-container",
    )

    # Patch `open` so that file access succeeds and the error comes from upload_blob.
    class FakeFile:
        def __init__(self, path: str):
            self.path = path

        def read(self, *_a, **_k):
            return b"fake-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def fake_open(path: str, mode: str = "r", *_, **__):
        return FakeFile(path)

    monkeypatch.setattr(services, "open", fake_open, raising=False)

    ctx: services.IngestionContext = {
        "status": "started",
        "raw_container_name": "raw-container",
        "integration_type": "results",
        "integration_provider": "kaggle",
        "integration_dataset": "owner/dataset",
        "local_integration_file_path": "/tmp/results.csv",
    }

    result = services.WriteRawSnapshotsToBlobStep()(ctx)

    assert result["status"] == "failed"
    assert "upload failed" in result["error_message"]

# Does LogIngestionEventStartStep call start_ingestion_event and pass the context to the next step?
# must call start_ingestion_event with the correct batch id, system event id, container name, integration type, integration provider and status and return the correct ingestion event id
def test_log_ingestion_event_start_success():
    class FakeSqlClient:
        def __init__(self):
            self.called_with = None

        def start_ingestion_event(self, **kwargs):
            self.called_with = kwargs
            return UUID("12345678-1234-5678-1234-567812345678")

    sql_client = FakeSqlClient()

    ctx: services.IngestionContext = {
        "sql_client": sql_client,
        "system_event_id": uuid4(),
        "raw_container_name": "raw",
        "integration_type": "results",
        "integration_provider": "kaggle",
        "status": "started",
    }

    result = services.LogIngestionEventStartStep()(ctx)

    assert "batch_id" in result
    assert isinstance(result["batch_id"], UUID)
    assert result["ingestion_event_id"] == UUID("12345678-1234-5678-1234-567812345678")
    assert sql_client.called_with["container_name"] == "raw"
    assert sql_client.called_with["integration_type"] == "results"

# Does LogIngestionEventStartStep handle errors gracefully?
# must not crash the pipeline, must set the status to "failed" and set the error message
def test_log_ingestion_event_start_error():
    class FakeSqlClient:
        def start_ingestion_event(self, **_kwargs):
            raise RuntimeError("db error")

    ctx: services.IngestionContext = {
        "sql_client": FakeSqlClient(),
        "system_event_id": uuid4(),
        "raw_container_name": "raw",
        "integration_type": "results",
        "integration_provider": "kaggle",
        "status": "started",
    }

    with pytest.raises(RuntimeError):
        services.LogIngestionEventStartStep()(ctx)


# Does LogIngestionEventCompleteStep call update_ingestion_event and pass the context to the next step?
# must call update_ingestion_event with the correct ingestion event id, status, blob path and error message and return the correct context
def test_log_ingestion_event_complete_success():
    class FakeSqlClient:
        def __init__(self):
            self.updated_with = None

        def update_ingestion_event(self, **kwargs):
            self.updated_with = kwargs

    sql_client = FakeSqlClient()

    ctx: services.IngestionContext = {
        "sql_client": sql_client,
        "ingestion_event_id": UUID("12345678-1234-5678-1234-567812345678"),
        "status": "ingested",
        "blob_snapshot_uri": "results/kaggle/dataset.csv",
    }

    result = services.LogIngestionEventCompleteStep()(ctx)

    assert result is ctx
    assert sql_client.updated_with["ingestion_event_id"] == UUID(
        "12345678-1234-5678-1234-567812345678"
    )
    assert sql_client.updated_with["status"] == "ingested"
    assert sql_client.updated_with["blob_path"] == "results/kaggle/dataset.csv"
    assert sql_client.updated_with["error_message"] is None


# Does LogIngestionEventCompleteStep handle errors gracefully?
# must not crash the pipeline, must set the status to "failed" and set the error message
def test_log_ingestion_event_complete_failure():
    class FakeSqlClient:
        def __init__(self):
            self.updated_with = None

        def update_ingestion_event(self, **kwargs):
            self.updated_with = kwargs

    sql_client = FakeSqlClient()

    ctx: services.IngestionContext = {
        "sql_client": sql_client,
        "ingestion_event_id": UUID("12345678-1234-5678-1234-567812345678"),
        "status": "failed",
        "blob_snapshot_uri": "results/kaggle/dataset.csv",
        "error_message": "something went wrong",
    }

    with pytest.raises(Exception) as exc:
        services.LogIngestionEventCompleteStep()(ctx)

    assert "something went wrong" in str(exc.value)
    assert sql_client.updated_with["status"] == "failed"
    assert sql_client.updated_with["blob_path"] == "results/kaggle/dataset.csv"
    assert sql_client.updated_with["error_message"] == "something went wrong"

# Does run_ingestion use the correct pipeline and pass the context to the pipeline?
# must use the correct pipeline and pass the context to the pipeline and return the correct context
def test_run_ingestion_uses_pipeline(monkeypatch):
    captured = {}

    class DummyPipeline:
        def __init__(self):
            self.context = None

        def run(self, context: services.IngestionContext) -> services.IngestionContext:
            self.context = context
            context["touched_by_pipeline"] = True
            return context

    def fake_build_pipeline_for(name: str):
        assert name == "test_pipeline"
        pipeline = DummyPipeline()
        captured["pipeline"] = pipeline
        return pipeline

    monkeypatch.setattr(services, "build_pipeline_for", fake_build_pipeline_for)

    class FakeSqlClient:
        pass

    system_event_id = uuid4()

    services.run_ingestion(
        pipeline_name="test_pipeline",
        sql_client=FakeSqlClient(),
        system_event_id=system_event_id,
        integration_type="results",
        integration_provider="kaggle",
    )

    pipeline = captured["pipeline"]
    assert pipeline.context["system_event_id"] == system_event_id
    assert pipeline.context["integration_type"] == "results"
    assert pipeline.context["integration_provider"] == "kaggle"
    assert pipeline.context["touched_by_pipeline"] is True

# Does public ingest functions delegate to run_ingestion?
# must delegate to run_ingestion with the correct pipeline name, sql client, system event id, integration type and integration provider
def test_public_ingest_functions_delegate_to_run_ingestion(monkeypatch):
    calls = []

    def fake_run_ingestion(
        *, pipeline_name: str, sql_client, system_event_id, integration_type: str, integration_provider: str
    ):
        calls.append(
            {
                "pipeline_name": pipeline_name,
                "sql_client": sql_client,
                "system_event_id": system_event_id,
                "integration_type": integration_type,
                "integration_provider": integration_provider,
            }
        )

    monkeypatch.setattr(services, "run_ingestion", fake_run_ingestion)

    class FakeSqlClient:
        pass

    sql_client = FakeSqlClient()
    system_event_id = uuid4()

    services.ingest_historical_kaggle_results(sql_client, system_event_id)
    services.ingest_rugby365_results(sql_client, system_event_id)
    services.ingest_rugby365_fixtures(sql_client, system_event_id)

    assert {call["pipeline_name"] for call in calls} == {
        "kaggle_historical_results",
        "rugby365_results",
        "rugby365_fixtures",
    }
    for call in calls:
        assert call["sql_client"] is sql_client
        assert call["system_event_id"] == system_event_id


