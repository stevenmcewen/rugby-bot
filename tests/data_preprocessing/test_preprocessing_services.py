from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from functions.data_preprocessing import preprocessing_services as services

# create a dummy step for testing
class DummyStep:
    def __init__(self, name: str):
        self.name = name
        self.called_with = []

    def __call__(self, context: services.PreprocessingContext) -> services.PreprocessingContext:
        self.called_with.append(dict(context))
        context[f"visited_{self.name}"] = True
        return context

# Does the preprocessing pipeline run the steps in order?
def test_preprocessing_pipeline_runs_steps_in_order():
    step1 = DummyStep("one")
    step2 = DummyStep("two")
    pipeline = services.PreprocessingPipeline(step1, step2)

    ctx = {"initial": True}
    out = pipeline.run(ctx)

    assert out["initial"] is True
    assert out["visited_one"] is True
    assert out["visited_two"] is True
    assert len(step1.called_with) == 1
    assert len(step2.called_with) == 1

# Does the register_preprocessing_pipeline function raise a ValueError if the pipeline name is already registered?
def test_register_preprocessing_pipeline_duplicate_raises(monkeypatch):
    monkeypatch.setattr(services, "PREPROCESSING_PIPELINE_REGISTRY", {}, raising=True)

    services.register_preprocessing_pipeline("x", lambda: ())
    with pytest.raises(ValueError):
        services.register_preprocessing_pipeline("x", lambda: ())

# Does the build_preprocessing_pipeline_for function raise a ValueError if the pipeline name is not registered?
def test_build_preprocessing_pipeline_for_unknown_raises(monkeypatch):
    monkeypatch.setattr(services, "PREPROCESSING_PIPELINE_REGISTRY", {}, raising=True)
    with pytest.raises(ValueError):
        services.build_preprocessing_pipeline_for("missing")

# Does the orchestrate_preprocessing function build the pipeline and run it?
def test_orchestrate_preprocessing_builds_pipeline_and_runs(monkeypatch):
    captured = {"pipeline_name": None, "context": None}

    class FakePipeline:
        def run(self, context):
            captured["context"] = context
            context["ran"] = True
            return context

    def fake_build(name: str):
        captured["pipeline_name"] = name
        return FakePipeline()

    monkeypatch.setattr(services, "build_preprocessing_pipeline_for", fake_build, raising=True)

    sql = SimpleNamespace()
    system_event_id = uuid4()
    services.orchestrate_preprocessing(sql_client=sql, system_event_id=system_event_id, pipeline_name="p")

    assert captured["pipeline_name"] == "p"
    assert captured["context"]["sql_client"] is sql
    assert captured["context"]["system_event_id"] == system_event_id
    assert captured["context"]["status"] == "started"
    assert captured["context"]["ran"] is True

# Does the select_ingested_sources_step function create the ingestion sources?
def test_select_ingested_sources_step_creates_ingestion_sources():
    row1 = SimpleNamespace(
        id=uuid4(),
        batch_id=uuid4(),
        system_event_id=uuid4(),
        integration_type="results",
        integration_provider="kaggle",
        container_name="raw",
        blob_path="a.csv",
    )
    row2 = SimpleNamespace(
        id=uuid4(),
        batch_id=uuid4(),
        system_event_id=uuid4(),
        integration_type="fixtures",
        integration_provider="rugby365",
        container_name="raw",
        blob_path="b.csv",
    )

    class FakeSql:
        def get_ingestion_events_by_status(self, status: str):
            assert status == "ingested"
            return [row1, row2]

    ctx: services.PreprocessingContext = {"sql_client": FakeSql()}
    out = services.SelectIngestedSourcesStep()(ctx)

    assert "ingestion_sources" in out
    assert len(out["ingestion_sources"]) == 2
    assert out["ingestion_sources"][0].blob_path == "a.csv"

# Does the build_preprocessing_plans_step function create the preprocessing plans for each mapping?
def test_build_preprocessing_plans_step_creates_plans_for_each_mapping():
    ingestion_source = services.IngestionSource(
        id=uuid4(),
        batch_id=uuid4(),
        system_event_id=uuid4(),
        integration_type="results",
        integration_provider="kaggle",
        container_name="raw",
        blob_path="a.csv",
    )

    class FakeSql:
        def get_source_target_mapping(self, source_provider: str, source_type: str):
            assert source_provider == "kaggle"
            assert source_type == "results"
            return [
                {"target_table": "dbo.T1", "pipeline_name": "p1"},
                {"target_table": "dbo.T2", "pipeline_name": "p2"},
            ]

    ctx: services.PreprocessingContext = {
        "sql_client": FakeSql(),
        "ingestion_sources": [ingestion_source],
    }
    out = services.BuildPreprocessingPlansStep()(ctx)

    plans = out["preprocessing_plans"]
    assert len(plans) == 2
    assert {p.target_table for p in plans} == {"dbo.T1", "dbo.T2"}
    assert {p.pipeline_name for p in plans} == {"p1", "p2"}

# Does the persist_preprocessing_events_step function create the preprocessing events?
def test_persist_preprocessing_events_step_creates_preprocessing_events():
    plan = services.PreprocessingPlan(
        batch_id=uuid4(),
        system_event_id=uuid4(),
        integration_type="results",
        integration_provider="kaggle",
        container_name="raw",
        blob_path="a.csv",
        target_table="dbo.T1",
        pipeline_name="p1",
    )

    created_id = uuid4()

    class FakeSql:
        def create_preprocessing_event(self, preprocessing_plan):
            assert preprocessing_plan is plan
            return {
                "id": created_id,
                "batch_id": plan.batch_id,
                "system_event_id": plan.system_event_id,
                "integration_type": plan.integration_type,
                "integration_provider": plan.integration_provider,
                "container_name": plan.container_name,
                "blob_path": plan.blob_path,
                "target_table": plan.target_table,
                "pipeline_name": plan.pipeline_name,
                "status": "started",
                "error_message": None,
            }

    ctx: services.PreprocessingContext = {"sql_client": FakeSql(), "preprocessing_plans": [plan]}
    out = services.PersistPreprocessingEventsStep()(ctx)

    events = out["preprocessing_events"]
    assert len(events) == 1
    assert events[0].id == created_id
    assert events[0].pipeline_name == "p1"

# Does the run_preprocessing_pipelines_step function update the events and ingestion event status?
def test_run_preprocessing_pipelines_step_updates_events_and_ingestion_event_status(monkeypatch):
    batch_id = uuid4()
    ingestion_source = services.IngestionSource(
        id=uuid4(),
        batch_id=batch_id,
        system_event_id=uuid4(),
        integration_type="results",
        integration_provider="kaggle",
        container_name="raw",
        blob_path="a.csv",
    )

    ev1 = services.PreprocessingEvent(
        id=uuid4(),
        batch_id=batch_id,
        system_event_id=ingestion_source.system_event_id,
        integration_type=ingestion_source.integration_type,
        integration_provider=ingestion_source.integration_provider,
        container_name=ingestion_source.container_name,
        blob_path=ingestion_source.blob_path,
        target_table="dbo.T1",
        pipeline_name="p1",
        status="started",
        error_message=None,
    )
    ev2 = services.PreprocessingEvent(
        id=uuid4(),
        batch_id=batch_id,
        system_event_id=ingestion_source.system_event_id,
        integration_type=ingestion_source.integration_type,
        integration_provider=ingestion_source.integration_provider,
        container_name=ingestion_source.container_name,
        blob_path=ingestion_source.blob_path,
        target_table="dbo.T2",
        pipeline_name="p2",
        status="started",
        error_message=None,
    )

    def fake_run_pipeline(preprocessing_event, sql_client):
        preprocessing_event.status = "succeeded"
        preprocessing_event.error_message = None

    monkeypatch.setattr(services, "run_preprocessing_pipeline", fake_run_pipeline, raising=True)

    calls = {"update_pre": [], "update_ing": []}

    class FakeSql:
        def update_preprocessing_event(self, **kwargs):
            calls["update_pre"].append(kwargs)

        def get_preprocessing_events_by_batch_id(self, batch_id):
            assert batch_id == ingestion_source.batch_id
            return [SimpleNamespace(status="succeeded"), SimpleNamespace(status="succeeded")]

        def update_ingestion_event(self, **kwargs):
            calls["update_ing"].append(kwargs)

    ctx: services.PreprocessingContext = {
        "sql_client": FakeSql(),
        "preprocessing_events": [ev1, ev2],
        "ingestion_sources": [ingestion_source],
    }

    services.RunPreprocessingPipelinesStep()(ctx)

    assert len(calls["update_pre"]) == 2
    assert {c["status"] for c in calls["update_pre"]} == {"succeeded"}
    assert calls["update_ing"][0]["ingestion_event_id"] == ingestion_source.id
    assert calls["update_ing"][0]["status"] == "preprocessed"

# Does the run_preprocessing_pipelines_step function set the ingestion event as failed when any preprocessing failed?
def test_run_preprocessing_pipelines_step_sets_ingestion_failed_when_any_preprocessing_failed(monkeypatch):
    batch_id = uuid4()
    ingestion_source = services.IngestionSource(
        id=uuid4(),
        batch_id=batch_id,
        system_event_id=uuid4(),
        integration_type="results",
        integration_provider="kaggle",
        container_name="raw",
        blob_path="a.csv",
    )

    ev = services.PreprocessingEvent(
        id=uuid4(),
        batch_id=batch_id,
        system_event_id=ingestion_source.system_event_id,
        integration_type=ingestion_source.integration_type,
        integration_provider=ingestion_source.integration_provider,
        container_name=ingestion_source.container_name,
        blob_path=ingestion_source.blob_path,
        target_table="dbo.T1",
        pipeline_name="p1",
        status="started",
        error_message=None,
    )

    def fake_run_pipeline(preprocessing_event, sql_client):
        preprocessing_event.status = "failed"
        preprocessing_event.error_message = "boom"

    monkeypatch.setattr(services, "run_preprocessing_pipeline", fake_run_pipeline, raising=True)

    calls = {"update_ing": []}

    class FakeSql:
        def update_preprocessing_event(self, **_kwargs):
            pass

        def get_preprocessing_events_by_batch_id(self, batch_id):
            assert batch_id == ingestion_source.batch_id
            return [SimpleNamespace(status="failed")]

        def update_ingestion_event(self, **kwargs):
            calls["update_ing"].append(kwargs)

    ctx: services.PreprocessingContext = {
        "sql_client": FakeSql(),
        "preprocessing_events": [ev],
        "ingestion_sources": [ingestion_source],
    }

    services.RunPreprocessingPipelinesStep()(ctx)

    assert calls["update_ing"][0]["status"] == "failed to preprocess"
    assert "preprocessing failed" in calls["update_ing"][0]["error_message"].lower()
