from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
import function_app

# Set up a fake sql client for testing
class FakeSqlClient:
    def __init__(self):
        self.started_events = []
        self.completed_events = []

    def start_system_event(self, **kwargs):
        event_id = uuid4()
        self.started_events.append({"id": event_id, **kwargs})
        # The real client returns a SystemEvent object with an id attribute.
        return SimpleNamespace(id=event_id)

    def complete_system_event(self, **kwargs):
        self.completed_events.append(kwargs)


# Set up a fake queue message for testing
class FakeQueueMessage:
    def __init__(self, body: bytes):
        self._body = body

    def get_body(self) -> bytes:
        return self._body

# Set up a fake module with a fake sql client for testing
def _make_module_with_fake_sql_client(monkeypatch) -> FakeSqlClient:
    fake_sql = FakeSqlClient()
    monkeypatch.setattr(function_app, "sql_client", fake_sql)
    return fake_sql

# Does IngestHistoricalKaggleResults function call ingest_historical_kaggle_results and return the correct response?
# must call ingest_historical_kaggle_results with the correct sql client and system event id and return the correct response
def test_ingest_historical_kaggle_results_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_ingest_historical_kaggle_results(sql_client, system_event_id):
        calls["sql_client"] = sql_client
        calls["system_event_id"] = system_event_id

    monkeypatch.setattr(
        function_app,
        "ingest_historical_kaggle_results",
        fake_ingest_historical_kaggle_results,
    )

    # Act
    response = function_app.IngestHistoricalKaggleResults(req=object())

    # Assert HTTP response
    assert response.status_code == 200
    payload = json.loads(response.get_body().decode("utf-8"))
    assert payload["status"] == "ok"
    assert "system_event_id" in payload

    # Assert ingestion call wiring
    assert calls["sql_client"] is fake_sql
    assert str(calls["system_event_id"]) == payload["system_event_id"]

    # Best-effort completion should have been attempted with status=succeeded
    assert any(c["status"] == "succeeded" for c in fake_sql.completed_events)


# Does IngestHistoricalKaggleResults function handle ingestion failure gracefully?
# must handle ingestion failure gracefully and return the correct response and attempt to complete the system event with status=failed and set the correct error message
def test_ingest_historical_kaggle_results_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_ingest(*_args, **_kwargs):
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(
        function_app,
        "ingest_historical_kaggle_results",
        failing_ingest,
    )

    response = function_app.IngestHistoricalKaggleResults(req=object())

    assert response.status_code == 500
    payload = json.loads(response.get_body().decode("utf-8"))
    assert payload["status"] == "error"
    assert "system_event_id" in payload

    # Failed completion should have been attempted
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    # Error details should contain our message
    assert any("ingest failed" in (c.get("details") or "") for c in fake_sql.completed_events)


# Does IngestRugby365ResultsFunction call ingest_rugby365_results and complete the system event?
# must call ingest_rugby365_results with the correct sql client and system event id and mark the event as succeeded
def test_ingest_rugby365_results_timer_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_ingest_rugby365_results(sql_client, system_event_id):
        calls["sql_client"] = sql_client
        calls["system_event_id"] = system_event_id

    monkeypatch.setattr(
        function_app,
        "ingest_rugby365_results",
        fake_ingest_rugby365_results,
    )

    # Act: timer payload is unused by the function, so we can pass a simple object.
    result = function_app.IngestRugby365ResultsFunction(timer=object())

    # Timer functions don't return an HttpResponse, so just assert wiring and side effects.
    assert calls["sql_client"] is fake_sql
    assert any(
        c["system_event_id"] == calls["system_event_id"] and c["status"] == "succeeded"
        for c in fake_sql.completed_events
    )
    # Function should return None implicitly.
    assert result is None


# Does IngestRugby365ResultsFunction handle ingestion failures by marking the system event as failed?
# must set the system event status to failed and record the error details
def test_ingest_rugby365_results_timer_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_ingest(*_args, **_kwargs):
        raise RuntimeError("r365 ingest failed")
 
    monkeypatch.setattr(
        function_app,
        "ingest_rugby365_results",
        failing_ingest,
    )

    # Act
    result = function_app.IngestRugby365ResultsFunction(timer=object())

    # The timer function does not raise; it logs and marks the system event as failed.
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    assert any(
        "r365 ingest failed" in (c.get("details") or "") for c in fake_sql.completed_events
    )
    assert result is None


# Does IngestRugby365FixturesFunction call ingest_rugby365_fixtures and complete the system event?
# must call ingest_rugby365_fixtures with the correct sql client and system event id and mark the event as succeeded
def test_ingest_rugby365_fixtures_timer_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_ingest_rugby365_fixtures(sql_client, system_event_id):
        calls["sql_client"] = sql_client
        calls["system_event_id"] = system_event_id

    monkeypatch.setattr(
        function_app,
        "ingest_rugby365_fixtures",
        fake_ingest_rugby365_fixtures,
    )

    # Act: timer payload is unused by the function, so we can pass a simple object.
    result = function_app.IngestRugby365FixturesFunction(timer=object())

    # Timer functions don't return an HttpResponse, so just assert wiring and side effects.
    assert calls["sql_client"] is fake_sql
    assert any(
        c["system_event_id"] == calls["system_event_id"] and c["status"] == "succeeded"
        for c in fake_sql.completed_events
    )
    # Function should return None implicitly.
    assert result is None


# Does IngestRugby365FixturesFunction handle ingestion failures by marking the system event as failed and re-raising?
# must set the system event status to failed, record the error details, and re-raise the original exception
def test_ingest_rugby365_fixtures_timer_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_ingest(*_args, **_kwargs):
        raise RuntimeError("r365 fixtures ingest failed")

    monkeypatch.setattr(
        function_app,
        "ingest_rugby365_fixtures",
        failing_ingest,
    )

    # The fixtures function logs, marks the system event as failed, and then re-raises.
    with pytest.raises(RuntimeError) as exc:
        function_app.IngestRugby365FixturesFunction(timer=object())

    assert "r365 fixtures ingest failed" in str(exc.value)
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    assert any(
        "r365 fixtures ingest failed" in (c.get("details") or "") for c in fake_sql.completed_events
    )


# Does PreprocessDataFunction call orchestrate_preprocessing and complete the system event?
# must call orchestrate_preprocessing with the correct sql client and system event id and mark the event as succeeded
def test_preprocess_data_timer_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_orchestrate_preprocessing(*, sql_client, system_event_id, pipeline_name="default_preprocessing"):
        calls["sql_client"] = sql_client
        calls["system_event_id"] = system_event_id
        calls["pipeline_name"] = pipeline_name

    monkeypatch.setattr(function_app, "orchestrate_preprocessing", fake_orchestrate_preprocessing)

    result = function_app.PreprocessDataFunction(timer=object())

    assert calls["sql_client"] is fake_sql
    assert calls["pipeline_name"] == "default_preprocessing"
    assert any(
        c["system_event_id"] == calls["system_event_id"] and c["status"] == "succeeded"
        for c in fake_sql.completed_events
    )
    assert result is None


# Does PreprocessDataFunction handle preprocessing failures by marking the system event as failed and re-raising?
# must set the system event status to failed, record the error details, and re-raise the original exception
def test_preprocess_data_timer_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_orchestrate(*_args, **_kwargs):
        raise RuntimeError("preprocess failed")

    monkeypatch.setattr(function_app, "orchestrate_preprocessing", failing_orchestrate)

    with pytest.raises(RuntimeError) as exc:
        function_app.PreprocessDataFunction(timer=object())

    assert "preprocess failed" in str(exc.value)
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    assert any(
        "preprocess failed" in (c.get("details") or "") for c in fake_sql.completed_events
    )


# Does TrainInternationalRugbyFixturesModelFunction call orchestrate_model_training and complete the system event?
# must call orchestrate_model_training with the correct parameters and mark the event as succeeded
def test_train_international_rugby_fixtures_model_timer_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_orchestrate_model_training(*, sql_client, system_event_id, pipeline_name, model_group_key):
        calls["sql_client"] = sql_client
        calls["system_event_id"] = system_event_id
        calls["pipeline_name"] = pipeline_name
        calls["model_group_key"] = model_group_key

    monkeypatch.setattr(function_app, "orchestrate_model_training", fake_orchestrate_model_training)

    result = function_app.TrainInternationalRugbyFixturesModelFunction(timer=object())

    assert calls["sql_client"] is fake_sql
    assert calls["pipeline_name"] == "default_model_training"
    assert calls["model_group_key"] == "international_rugby_fixtures"
    assert any(
        c["system_event_id"] == calls["system_event_id"] and c["status"] == "succeeded"
        for c in fake_sql.completed_events
    )
    assert result is None


# Does TrainInternationalRugbyFixturesModelFunction handle training failures by marking the system event as failed and re-raising?
# must set the system event status to failed, record the error details, and re-raise the original exception
def test_train_international_rugby_fixtures_model_timer_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_orchestrate(*_args, **_kwargs):
        raise RuntimeError("model training failed")

    monkeypatch.setattr(function_app, "orchestrate_model_training", failing_orchestrate)

    with pytest.raises(RuntimeError) as exc:
        function_app.TrainInternationalRugbyFixturesModelFunction(timer=object())

    assert "model training failed" in str(exc.value)
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    assert any(
        "model training failed" in (c.get("details") or "") for c in fake_sql.completed_events
    )


# Does ScoreUpcomingInternationalRugbyFixturesFunction call orchestrate_model_scoring and complete the system event?
# must call orchestrate_model_scoring with the correct parameters and mark the event as succeeded
def test_score_upcoming_international_rugby_fixtures_timer_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_orchestrate_model_scoring(*, sql_client, system_event_id, pipeline_name, model_group_key):
        calls["sql_client"] = sql_client
        calls["system_event_id"] = system_event_id
        calls["pipeline_name"] = pipeline_name
        calls["model_group_key"] = model_group_key

    monkeypatch.setattr(function_app, "orchestrate_model_scoring", fake_orchestrate_model_scoring)

    result = function_app.ScoreUpcomingInternationalRugbyFixturesFunction(timer=object())

    assert calls["sql_client"] is fake_sql
    assert calls["pipeline_name"] == "default_model_scoring"
    assert calls["model_group_key"] == "international_rugby_fixtures"
    assert any(
        c["system_event_id"] == calls["system_event_id"] and c["status"] == "succeeded"
        for c in fake_sql.completed_events
    )
    assert result is None


# Does ScoreUpcomingInternationalRugbyFixturesFunction handle scoring failures by marking the system event as failed and re-raising?
# must set the system event status to failed, record the error details, and re-raise the original exception
def test_score_upcoming_international_rugby_fixtures_timer_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_orchestrate(*_args, **_kwargs):
        raise RuntimeError("model scoring failed")

    monkeypatch.setattr(function_app, "orchestrate_model_scoring", failing_orchestrate)

    with pytest.raises(RuntimeError) as exc:
        function_app.ScoreUpcomingInternationalRugbyFixturesFunction(timer=object())

    assert "model scoring failed" in str(exc.value)
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    assert any(
        "model scoring failed" in (c.get("details") or "") for c in fake_sql.completed_events
    )


# Does DailyPredictionsNotificationFunction call get_daily_predictions and send_prediction_email and complete the system event?
# must call get_daily_predictions and send_prediction_email with the correct parameters and mark the event as succeeded
def test_daily_predictions_notification_timer_success(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    calls = {}

    def fake_get_daily_predictions(*, sql_client):
        calls["get_daily_predictions_sql_client"] = sql_client
        return {"date": "2026-01-15", "predictions": {}}

    def fake_send_prediction_email(*, payload):
        calls["send_prediction_email_payload"] = payload

    monkeypatch.setattr(function_app, "get_daily_predictions", fake_get_daily_predictions)
    monkeypatch.setattr(function_app, "send_prediction_email", fake_send_prediction_email)

    result = function_app.DailyPredictionsNotificationFunction(timer=object())

    assert calls["get_daily_predictions_sql_client"] is fake_sql
    assert calls["send_prediction_email_payload"] == {"date": "2026-01-15", "predictions": {}}
    assert any(c["status"] == "succeeded" for c in fake_sql.completed_events)
    assert result is None


# Does DailyPredictionsNotificationFunction handle notification failures by marking the system event as failed and re-raising?
# must set the system event status to failed, record the error details, and re-raise the original exception
def test_daily_predictions_notification_timer_failure(monkeypatch):
    fake_sql = _make_module_with_fake_sql_client(monkeypatch)

    def failing_get_predictions(*_args, **_kwargs):
        raise RuntimeError("predictions generation failed")

    monkeypatch.setattr(function_app, "get_daily_predictions", failing_get_predictions)

    with pytest.raises(RuntimeError) as exc:
        function_app.DailyPredictionsNotificationFunction(timer=object())

    assert "predictions generation failed" in str(exc.value)
    assert any(c["status"] == "failed" for c in fake_sql.completed_events)
    assert any(
        "predictions generation failed" in (c.get("details") or "") for c in fake_sql.completed_events
    )