from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from functions.data_preprocessing import preprocessing_pipelines as pipes

# create a dummy event for testing
class DummyEvent:
    def __init__(self, pipeline_name: str):
        self.id = uuid4()
        self.pipeline_name = pipeline_name
        self.status = "started"
        self.error_message = None
        self.container_name = "raw"
        self.blob_path = "x.csv"
        self.integration_provider = "kaggle"
        self.integration_type = "historical_results"
        self.target_table = "dbo.Target"

# Does the register_preprocessing_pipeline function raise a ValueError if the pipeline name is already registered?
def test_register_preprocessing_pipeline_duplicate_raises(monkeypatch):
    # isolate registry for this test
    registry = {}
    monkeypatch.setattr(pipes, "PREPROCESSING_HANDLER_REGISTRY", registry, raising=True)

    def handler(_event, _sql):
        return None

    pipes.register_preprocessing_pipeline("x", handler)
    with pytest.raises(ValueError):
        pipes.register_preprocessing_pipeline("x", handler)

# Does the resolve_preprocessing_handler function raise a ValueError if the pipeline name is not registered?
def test_resolve_preprocessing_handler_unknown_raises(monkeypatch):
    monkeypatch.setattr(pipes, "PREPROCESSING_HANDLER_REGISTRY", {}, raising=True)
    with pytest.raises(ValueError):
        pipes._resolve_preprocessing_handler("missing")

# Does the run_preprocessing_pipeline function mark the event as running and succeed?
def test_run_preprocessing_pipeline_marks_running_and_succeeds(monkeypatch):
    event = DummyEvent("ok")

    calls = {"updated": [], "ran": 0}

    def fake_update_preprocessing_event(*, preprocessing_event_id, status, error_message=None):
        calls["updated"].append((preprocessing_event_id, status, error_message))

    def handler(e, sql_client):
        assert e is event
        assert sql_client is not None
        calls["ran"] += 1

    monkeypatch.setattr(pipes, "PREPROCESSING_HANDLER_REGISTRY", {"ok": handler}, raising=True)

    sql = SimpleNamespace(update_preprocessing_event=fake_update_preprocessing_event)

    pipes.run_preprocessing_pipeline(event, sql)

    assert calls["updated"][0][0] == event.id
    assert calls["updated"][0][1] == "running"
    assert calls["ran"] == 1
    assert event.status == "succeeded"
    assert event.error_message is None

# Does the run_preprocessing_pipeline function set the event as failed when the handler is missing?
def test_run_preprocessing_pipeline_sets_failed_when_handler_missing(monkeypatch):
    event = DummyEvent("missing")
    monkeypatch.setattr(pipes, "PREPROCESSING_HANDLER_REGISTRY", {}, raising=True)

    sql = SimpleNamespace(update_preprocessing_event=lambda **_k: None)

    pipes.run_preprocessing_pipeline(event, sql)

    assert event.status == "failed"
    assert "no preprocessing handler registered" in (event.error_message or "").lower()

# Does the run_preprocessing_pipeline function set the event as failed when the handler raises an exception?
def test_run_preprocessing_pipeline_sets_failed_when_handler_raises(monkeypatch):
    event = DummyEvent("boom")

    def handler(_event, _sql):
        raise RuntimeError("explode")

    monkeypatch.setattr(pipes, "PREPROCESSING_HANDLER_REGISTRY", {"boom": handler}, raising=True)

    sql = SimpleNamespace(update_preprocessing_event=lambda **_k: None)

    pipes.run_preprocessing_pipeline(event, sql)

    assert event.status == "failed"
    assert "explode" in (event.error_message or "")

# Does the historical_kaggle_international_results_preprocessing_pipeline function do nothing if the source data is empty?
def test_historical_kaggle_pipeline_no_source_rows_is_noop(monkeypatch):
    event = DummyEvent("historical_kaggle_international_results_preprocessing")

    called = {"validate_source": 0, "write": 0}

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(pipes, "get_source_schema", lambda *_a, **_k: {}, raising=True)
    monkeypatch.setattr(pipes, "get_target_schema", lambda *_a, **_k: {}, raising=True)

    def fake_validate_source(*_a, **_k):
        called["validate_source"] += 1

    def fake_write(*_a, **_k):
        called["write"] += 1

    monkeypatch.setattr(pipes, "validate_source_data", fake_validate_source, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.historical_kaggle_international_results_preprocessing_pipeline(event, sql_client=SimpleNamespace())
    assert called["validate_source"] == 0
    assert called["write"] == 0

# Does the historical_kaggle_international_results_preprocessing_pipeline function call the validators and write the data to the target table?
def test_historical_kaggle_pipeline_happy_path_calls_validators_and_write(monkeypatch):
    event = DummyEvent("historical_kaggle_international_results_preprocessing")
    called = {"source": 0, "validate_source": 0, "validate_trans": 0, "write": 0}
    src = pd.DataFrame([{"a": 1}])
    out = pd.DataFrame([{"b": 2}])
    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: src, raising=True)
    monkeypatch.setattr(pipes, "get_source_schema", lambda *_a, **_k: {"columns": []}, raising=True)
    monkeypatch.setattr(pipes, "get_target_schema", lambda *_a, **_k: {"columns": []}, raising=True)
    def fake_validate_source(*_a, **_k):
        called["validate_source"] += 1
    def fake_transform(*_a, **_k):
        return out
    def fake_validate_trans(*_a, **_k):
        called["validate_trans"] += 1
    def fake_write(*_a, **_k):
        called["write"] += 1
    monkeypatch.setattr(pipes, "validate_source_data", fake_validate_source, raising=True)
    monkeypatch.setattr(pipes, "transform_kaggle_historical_data_to_international_results", fake_transform, raising=True)
    monkeypatch.setattr(pipes, "validate_transformed_data", fake_validate_trans, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)
    pipes.historical_kaggle_international_results_preprocessing_pipeline(event, sql_client=SimpleNamespace())
    assert called["validate_source"] == 1
    assert called["validate_trans"] == 1
    assert called["write"] == 1

# Does the rugby365_international_results_preprocessing_pipeline function call the validators and write the data to the target table?
def test_rugby365_results_pipeline_happy_path_calls_validators_and_write(monkeypatch):
    event = DummyEvent("rugby365_international_results_preprocessing")

    called = {"source": 0, "validate_source": 0, "validate_trans": 0, "write": 0}

    src = pd.DataFrame([{"a": 1}])
    out = pd.DataFrame([{"b": 2}])

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: src, raising=True)
    monkeypatch.setattr(pipes, "get_source_schema", lambda *_a, **_k: {"columns": []}, raising=True)
    monkeypatch.setattr(pipes, "get_target_schema", lambda *_a, **_k: {"columns": []}, raising=True)

    def fake_validate_source(*_a, **_k):
        called["validate_source"] += 1

    def fake_transform(*_a, **_k):
        return out

    def fake_validate_trans(*_a, **_k):
        called["validate_trans"] += 1

    def fake_write(*_a, **_k):
        called["write"] += 1

    monkeypatch.setattr(pipes, "validate_source_data", fake_validate_source, raising=True)
    monkeypatch.setattr(pipes, "transform_rugby365_results_data_to_international_results", fake_transform, raising=True)
    monkeypatch.setattr(pipes, "validate_transformed_data", fake_validate_trans, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.rugby365_international_results_preprocessing_pipeline(event, sql_client=SimpleNamespace())

    assert called["validate_source"] == 1
    assert called["validate_trans"] == 1
    assert called["write"] == 1

# Does the rugby365_international_fixtures_preprocessing_pipeline function call the validators and write the data to the target table?
def test_rugby365_fixtures_pipeline_happy_path_calls_validators_and_write(monkeypatch):
    event = DummyEvent("rugby365_international_fixtures_preprocessing")

    called = {"source": 0, "validate_source": 0, "validate_trans": 0, "truncate": 0, "write": 0}

    src = pd.DataFrame([{"a": 1}])
    out = pd.DataFrame([{"b": 2}])

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: src, raising=True)
    monkeypatch.setattr(pipes, "get_source_schema", lambda *_a, **_k: {"columns": []}, raising=True)
    monkeypatch.setattr(pipes, "get_target_schema", lambda *_a, **_k: {"columns": []}, raising=True)

    def fake_validate_source(*_a, **_k):
        called["validate_source"] += 1

    def fake_transform(*_a, **_k):
        return out

    def fake_validate_trans(*_a, **_k):
        called["validate_trans"] += 1

    def fake_truncate(*_a, **_k):
        called["truncate"] += 1

    def fake_write(*_a, **_k):
        called["write"] += 1

    monkeypatch.setattr(pipes, "validate_source_data", fake_validate_source, raising=True)
    monkeypatch.setattr(pipes, "transform_rugby365_fixtures_data_to_international_fixtures", fake_transform, raising=True)
    monkeypatch.setattr(pipes, "validate_transformed_data", fake_validate_trans, raising=True)
    monkeypatch.setattr(pipes, "truncate_target_table", fake_truncate, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.rugby365_international_fixtures_preprocessing_pipeline(event, sql_client=SimpleNamespace())

    assert called["validate_source"] == 1
    assert called["validate_trans"] == 1
    assert called["truncate"] == 1
    assert called["write"] == 1


def test_model_ready_pipeline_no_source_rows_is_noop(monkeypatch):
    event = DummyEvent("international_results_to_model_ready_data_preprocessing")

    called = {"truncate": 0, "write": 0}

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: pd.DataFrame(), raising=True)

    def fake_truncate(*_a, **_k):
        called["truncate"] += 1

    def fake_write(*_a, **_k):
        called["write"] += 1

    monkeypatch.setattr(pipes, "truncate_target_table", fake_truncate, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.international_results_to_model_ready_data_preprocessing_pipeline(event, sql_client=SimpleNamespace())
    assert called["truncate"] == 0
    assert called["write"] == 0


def test_model_ready_pipeline_happy_path_truncates_and_writes(monkeypatch):
    event = DummyEvent("international_results_to_model_ready_data_preprocessing")

    called = {"validate_source": 0, "validate_trans": 0, "truncate": 0, "write": 0}

    src = pd.DataFrame([{"a": 1}])
    out = pd.DataFrame([{"b": 2}])

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: src, raising=True)
    monkeypatch.setattr(pipes, "get_source_schema", lambda *_a, **_k: {"columns": []}, raising=True)
    monkeypatch.setattr(pipes, "get_target_schema", lambda *_a, **_k: {"columns": []}, raising=True)

    def fake_validate_source(*_a, **_k):
        called["validate_source"] += 1

    def fake_transform(*_a, **_k):
        return out

    def fake_validate_trans(*_a, **_k):
        called["validate_trans"] += 1

    def fake_truncate(*_a, **_k):
        called["truncate"] += 1

    def fake_write(*_a, **_k):
        called["write"] += 1

    monkeypatch.setattr(pipes, "validate_source_data", fake_validate_source, raising=True)
    monkeypatch.setattr(
        pipes,
        "transform_international_results_to_model_ready_data",
        fake_transform,
        raising=True,
    )
    monkeypatch.setattr(pipes, "validate_transformed_data", fake_validate_trans, raising=True)
    monkeypatch.setattr(pipes, "truncate_target_table", fake_truncate, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.international_results_to_model_ready_data_preprocessing_pipeline(event, sql_client=SimpleNamespace())

    assert called["validate_source"] == 1
    assert called["validate_trans"] == 1
    assert called["truncate"] == 1
    assert called["write"] == 1


def test_fixtures_model_ready_pipeline_no_source_rows_is_noop(monkeypatch):
    event = DummyEvent("international_fixtures_to_model_ready_data_preprocessing")

    called = {"truncate": 0, "write": 0}

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: pd.DataFrame(), raising=True)

    def fake_truncate(*_a, **_k):
        called["truncate"] += 1

    def fake_write(*_a, **_k):
        called["write"] += 1

    monkeypatch.setattr(pipes, "truncate_target_table", fake_truncate, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.international_fixtures_to_model_ready_data_preprocessing_pipeline(event, sql_client=SimpleNamespace())
    assert called["truncate"] == 0
    assert called["write"] == 0


def test_fixtures_model_ready_pipeline_happy_path_truncates_then_writes(monkeypatch):
    event = DummyEvent("international_fixtures_to_model_ready_data_preprocessing")

    called = {"validate_source": 0, "validate_trans": 0}
    call_order: list[str] = []

    src = pd.DataFrame([{"a": 1}])
    out = pd.DataFrame([{"KickoffTimeLocal": pd.Timestamp("2025-12-15 18:00:00")}])

    monkeypatch.setattr(pipes, "get_source_data", lambda *_a, **_k: src, raising=True)
    monkeypatch.setattr(pipes, "get_source_schema", lambda *_a, **_k: {"columns": []}, raising=True)
    monkeypatch.setattr(pipes, "get_target_schema", lambda *_a, **_k: {"columns": []}, raising=True)

    def fake_validate_source(*_a, **_k):
        called["validate_source"] += 1

    def fake_transform(*_a, **_k):
        return out

    def fake_validate_trans(*_a, **_k):
        called["validate_trans"] += 1

    def fake_truncate(*_a, **_k):
        call_order.append("truncate")

    def fake_write(*_a, **_k):
        call_order.append("write")

    monkeypatch.setattr(pipes, "validate_source_data", fake_validate_source, raising=True)
    monkeypatch.setattr(
        pipes,
        "transform_international_fixtures_to_model_ready_data",
        fake_transform,
        raising=True,
    )
    monkeypatch.setattr(pipes, "validate_transformed_data", fake_validate_trans, raising=True)
    monkeypatch.setattr(pipes, "truncate_target_table", fake_truncate, raising=True)
    monkeypatch.setattr(pipes, "write_data_to_target_table", fake_write, raising=True)

    pipes.international_fixtures_to_model_ready_data_preprocessing_pipeline(event, sql_client=SimpleNamespace())

    assert called["validate_source"] == 1
    assert called["validate_trans"] == 1
    assert call_order == ["truncate", "write"]