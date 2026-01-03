from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from functions.ml_models import ml_orchestrator as orch

# Tests for orchestrate_model_training 
# Does the pipeline get built and run with correct context?
def test_orchestrate_model_training_builds_pipeline_and_runs_context(monkeypatch):
    calls = {}

    class FakePipeline:
        def run(self, context):
            calls["context"] = context
            return context

    monkeypatch.setattr(orch, "build_model_pipeline_for", lambda name: FakePipeline())

    sql_client = object()
    system_event_id = uuid4()
    orch.orchestrate_model_training(
        sql_client=sql_client,
        system_event_id=system_event_id,
        pipeline_name="default_model_training",
        model_group_key="group1",
    )

    assert calls["context"]["sql_client"] is sql_client
    assert calls["context"]["system_event_id"] == system_event_id
    assert calls["context"]["model_group_key"] == "group1"
    assert calls["context"]["status"] == "started"

# Tests for orchestrate_model_scoring
# Does the pipeline get built and run with correct context?
def test_orchestrate_model_scoring_builds_pipeline_and_runs_context(monkeypatch):
    calls = {}

    class FakePipeline:
        def run(self, context):
            calls["context"] = context
            return context

    monkeypatch.setattr(orch, "build_model_pipeline_for", lambda name: FakePipeline())

    sql_client = object()
    system_event_id = uuid4()
    orch.orchestrate_model_scoring(
        sql_client=sql_client,
        system_event_id=system_event_id,
        pipeline_name="default_model_scoring",
        model_group_key="group2",
    )

    assert calls["context"]["sql_client"] is sql_client
    assert calls["context"]["system_event_id"] == system_event_id
    assert calls["context"]["model_group_key"] == "group2"
    assert calls["context"]["status"] == "started"


