from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from functions.ml_models import ml_pipelines as pipes

# Tests the build_model_pipeline_for function for the default training pipeline
# Does the pipeline contain the expected steps in the correct order for default training?
def test_build_model_pipeline_for_default_training_contains_expected_steps():
    pipeline = pipes.build_model_pipeline_for("default_model_training")
    step_names = [s.__class__.__name__ for s in pipeline._steps]  # type: ignore[attr-defined]
    assert step_names == [
        "StartTrainingRunStep",
        "LoadModelSpecStep",
        "LoadDataStep",
        "PrepareTrainingContextStep",
        "BuildTrainingPayloadsStep",
        "TrainModelsStep",
        "EvaluateModelsStep",
        "PersistModelArtifactsStep",
    ]

def test_build_model_pipeline_for_default_scoring_contains_expected_steps():
    pipeline = pipes.build_model_pipeline_for("default_model_scoring")
    step_names = [s.__class__.__name__ for s in pipeline._steps]  # type: ignore[attr-defined]
    assert step_names == [
        "StartScoringRunStep",
        "LoadModelSpecStep",
        "LoadDataStep",
        "PrepareScoringContextStep",
        "LoadModelArtifactsStep",
        "ScoreModelsStep",
        "CombineScoringResultsStep",
        "PersistScoringResultsStep",
    ]

# Tests the LoadDataStep to ensure it selects the correct columns for training
def test_load_data_step_selects_expected_columns_for_training(monkeypatch):
    captured = {}

    class FakeSqlClient:
        def get_model_source_data(self, *, source_table: str, columns_to_select: list[str]):
            captured["source_table"] = source_table
            captured["columns_to_select"] = columns_to_select
            # return a df containing selected columns
            return pd.DataFrame({c: [1] for c in columns_to_select})

    context = {
        "sql_client": FakeSqlClient(),
            "model_group_key": "group1",
        "run_type": "training",
        "model_specs": {
            "training_dataset_source": "dbo.Train",
            "scoring_dataset_source": "dbo.Score",
            "columns": {"entity": ["ID"], "feature": ["F1", "F2"], "weight": "W"},
            "models": [
                {"model_key": "m1", "trainer_key": "t", "target_column": "Y1", "prediction_type": "classification", "is_enabled": True, "sample_weight_column": None, "time_column": "T"},
                {"model_key": "m2", "trainer_key": "t", "target_column": "Y2", "prediction_type": "classification", "is_enabled": True, "sample_weight_column": "W2", "time_column": None},
            ],
        },
    }

    out = pipes.LoadDataStep()(context)
    assert captured["source_table"] == "dbo.Train"
    # order-preserving de-dupe of entity + features + targets + weights + time columns
    assert captured["columns_to_select"] == ["ID", "F1", "F2", "Y1", "Y2", "W", "W2", "T"]
    assert isinstance(out["data"], pd.DataFrame)

# Tests the StartTrainingRunStep to ensure it sets the initial status
def test_start_training_run_step_sets_initial_status():
    context = {"sql_client": object(), "system_event_id": "sys"}
    out = pipes.StartTrainingRunStep()(context)
    assert out["status"] == "started"
    assert out["run_type"] == "training"
    assert out["sql_client"] is context["sql_client"]
    assert out["system_event_id"] == "sys"

def test_start_scoring_run_step_sets_initial_status():
    ctx = {"sql_client": object(), "system_event_id": "sys"}
    out = pipes.StartScoringRunStep()(ctx)
    assert out["status"] == "started"
    assert out["run_type"] == "scoring"
    assert out["sql_client"] is ctx["sql_client"]
    assert out["system_event_id"] == "sys"

# Tests the PrepareTrainingContextStep to ensure it computes schema hash and validates columns
def test_prepare_training_context_computes_schema_hash_and_validates_columns(monkeypatch):
    monkeypatch.setattr(pipes, "stable_schema_hash", lambda cols: "HASH:" + ",".join(cols))

    data = pd.DataFrame({"F1": [1], "F2": [2], "Y": [0]})
    context = {
        "data": data,
        "model_specs": {
            "columns": {"feature": ["F1", "F2"], "entity": []},
            "models": [{"model_key": "m", "trainer_key": "t", "target_column": "Y", "prediction_type": "classification"}],
        },
    }

    out = pipes.PrepareTrainingContextStep()(context)
    assert out["resolved_schema_hash"] == "HASH:F1,F2"
    assert list(out["X_shared"].columns) == ["F1", "F2"]


# Tests the BuildTrainingPayloadsStep to ensure it uses group weight fallback and allows overrides
def test_build_training_payloads_uses_group_weight_fallback_and_allows_overrides():
    data = pd.DataFrame({"F1": [1], "Y": [0], "W": [0.5], "W2": [0.7], "T": ["2025-01-01"]})
    ctx = {
        "data": data,
        "model_specs": {
            "models": [
                {"model_key": "m1", "trainer_key": "t", "target_column": "Y", "prediction_type": "classification", "model_parameters": {}, "trainer_parameters": {}, "sample_weight_column": None, "time_column": "T"},
                {"model_key": "m2", "trainer_key": "t", "target_column": "Y", "prediction_type": "classification", "model_parameters": {}, "trainer_parameters": {}, "sample_weight_column": "W2", "time_column": None},
            ]
        },
        "resolved_feature_columns": ["F1"],
        "resolved_entity_columns": [],
        "resolved_group_weight_column": "W",
        "X_shared": data[["F1"]],
    }

    out = pipes.BuildTrainingPayloadsStep()(ctx)
    p1 = out["train_payloads"]["m1"]
    p2 = out["train_payloads"]["m2"]
    assert p1.sample_weight_column == "W"
    assert p2.sample_weight_column == "W2"
    assert p1.time_column == "T"
    assert p2.time_column is None

# Tests the TrainModelsStep to ensure it calls train_model and sets status
def test_train_models_step_calls_train_model_and_sets_status(monkeypatch):
    monkeypatch.setattr(pipes, "uct_now_iso", lambda: "NOW")
    monkeypatch.setattr(pipes, "train_model", lambda payload: f"trained:{payload.model_key}")

    payload = pipes.TrainPayload(
        model_key="m1",
        trainer_key="t",
        prediction_type="classification",
        entity_columns=[],
        feature_columns=["F1"],
        target_column="Y",
        sample_weight_column=None,
        time_column=None,
        X=pd.DataFrame({"F1": [1, 2]}),
        y=pd.Series([0, 1]),
        sample_weight=None,
        time=None,
        model_parameters={},
        trainer_parameters={},
    )

    ctx = {"resolved_schema_hash": "H", "train_payloads": {"m1": payload}}
    out = pipes.TrainModelsStep()(ctx)
    assert out["status"] == "trained"
    assert out["trained_models"]["m1"].model_object == "trained:m1"


# Tests the EvaluateModelsStep to ensure it calls evaluate_model and sets status
def test_evaluate_models_step_calls_evaluate_model(monkeypatch):
    monkeypatch.setattr(pipes, "uct_now_iso", lambda: "NOW")
    monkeypatch.setattr(pipes, "evaluate_model", lambda **kwargs: {"acc": 1.0})

    tm = pipes.TrainedModelResult(
        model_key="m1",
        trainer_key="t",
        prediction_type="classification",
        target_column="Y",
        feature_columns=["F1"],
        entity_columns=[],
        schema_hash="H",
        trained_at_utc="NOW",
        model_object=object(),
        train_rows=2,
    )
    payload = pipes.TrainPayload(
        model_key="m1",
        trainer_key="t",
        prediction_type="classification",
        entity_columns=[],
        feature_columns=["F1"],
        target_column="Y",
        sample_weight_column=None,
        time_column=None,
        X=pd.DataFrame({"F1": [1, 2]}),
        y=pd.Series([0, 1]),
        sample_weight=None,
        time=None,
        model_parameters={},
        trainer_parameters={},
    )

    ctx = {"trained_models": {"m1": tm}, "train_payloads": {"m1": payload}}
    out = pipes.EvaluateModelsStep()(ctx)
    assert out["status"] == "evaluated"
    assert out["evaluation_results"]["m1"].metrics == {"acc": 1.0}

# Tests the PersistModelArtifactsStep to ensure it serializes and persists model artifacts
def test_persist_model_artifacts_step_serializes_and_persists(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipes, "serialize_model_artifact", lambda obj: b"bytes")
    def fake_persist_model_artifact(**kwargs):
        calls["persist_kwargs"] = kwargs
        return ("id", 1)

    monkeypatch.setattr(pipes, "persist_model_artifact", fake_persist_model_artifact)

    tm = pipes.TrainedModelResult(
        model_key="m1",
        trainer_key="t",
        prediction_type="classification",
        target_column="Y",
        feature_columns=["F1"],
        entity_columns=[],
        schema_hash="H",
        trained_at_utc="NOW",
        model_object=object(),
        train_rows=2,
    )
    er = pipes.EvaluationResult(model_key="m1", metrics={"acc": 1.0}, evaluated_at_utc="NOW")
    ctx = {"sql_client": object(), "system_event_id": "sys", "trained_models": {"m1": tm}, "evaluation_results": {"m1": er}}

    out = pipes.PersistModelArtifactsStep()(ctx)
    assert out["status"] == "persisted"
    assert calls["persist_kwargs"]["model_key"] == "m1"
    assert calls["persist_kwargs"]["metrics"] == {"acc": 1.0}


# Tests the ModelPipeline to ensure it runs steps in order and passes context correctly
def test_model_pipeline_runs_steps_in_order_and_passes_context():
    calls = []

    class Step1:
        def __call__(self, context):
            calls.append("s1")
            context["a"] = 1
            return context

    class Step2:
        def __call__(self, context):
            calls.append("s2")
            context["b"] = context["a"] + 1
            return context

    pipeline = pipes.ModelPipeline(Step1(), Step2())
    out = pipeline.run({"x": 0})
    assert calls == ["s1", "s2"]
    assert out["a"] == 1
    assert out["b"] == 2


# Tests for build_model_pipeline_for function error handling
def test_build_model_pipeline_for_raises_on_unknown_name():
    with pytest.raises(ValueError, match="No model pipeline registered"):
        pipes.build_model_pipeline_for("does_not_exist")

# Tests for register_model_pipeline function to prevent duplicate registrations
def test_register_model_pipeline_raises_on_duplicate():
    name = "tmp_pipe_for_test"
    pipes.register_model_pipeline(name, lambda: tuple())
    try:
        with pytest.raises(ValueError, match="already registered"):
            pipes.register_model_pipeline(name, lambda: tuple())
    finally:
        # cleanup so test order doesn't matter
        pipes.MODEL_PIPELINE_REGISTRY.pop(name, None)

# Tests the LoadDataStep to ensure it raises on invalid run_type
def test_load_data_step_raises_on_invalid_run_type():
    class FakeSqlClient:
        def get_model_source_data(self, *, source_table: str, columns_to_select: list[str]):
            raise AssertionError("should not be called")

    context = {
        "sql_client": FakeSqlClient(),
        "model_group_key": "group1",
        "run_type": "nope",
        "model_specs": {"training_dataset_source": "dbo.Train", "scoring_dataset_source": "dbo.Score", "columns": {"feature": ["F1"]}, "models": []},
    }

    with pytest.raises(ValueError, match="Invalid run type"):
        pipes.LoadDataStep()(context)


def test_prepare_scoring_context_sets_index_and_shared_features(monkeypatch):
    monkeypatch.setattr(pipes, "stable_schema_hash", lambda cols: "HASH:" + ",".join(cols))
    data = pd.DataFrame(
        {
            "TeamId": [10, 20],
            "MatchId": [1, 2],
            "F1": [0.1, 0.2],
            "F2": [3, 4],
        }
    )
    ctx = {
        "data": data,
        "model_specs": {"columns": {"entity": ["TeamId", "MatchId"], "feature": ["F1", "F2"]}},
    }
    out = pipes.PrepareScoringContextStep()(ctx)
    assert out["status"] == "preparing_scoring"
    assert out["resolved_schema_hash"] == "HASH:F1,F2"
    assert out["resolved_entity_columns"] == ["TeamId", "MatchId"]
    assert out["resolved_feature_columns"] == ["F1", "F2"]
    assert list(out["X_shared"].columns) == ["F1", "F2"]
    # entity columns should now be the index (and still present as columns due to drop=False)
    assert list(out["data"].index.names) == ["TeamId", "MatchId"]
    assert "TeamId" in out["data"].columns


def test_prepare_scoring_context_raises_on_missing_columns():
    data = pd.DataFrame({"ID": [1], "F1": [1]})
    with pytest.raises(ValueError, match="Missing entity columns"):
        pipes.PrepareScoringContextStep()(
            {"data": data, "model_specs": {"columns": {"entity": ["MissingId"], "feature": ["F1"]}}}
        )
    with pytest.raises(ValueError, match="Missing feature columns"):
        pipes.PrepareScoringContextStep()(
            {"data": data, "model_specs": {"columns": {"entity": ["ID"], "feature": ["MissingF"]}}}
        )


def test_prepare_scoring_context_skips_when_data_is_empty():
    """Verify that scoring is gracefully skipped when there's no fixture data."""
    data = pd.DataFrame()  # Empty dataframe
    ctx = {
        "data": data,
        "model_group_key": "fixtures",
        "model_specs": {"columns": {"entity": ["ID"], "feature": ["F1"]}},
    }
    out = pipes.PrepareScoringContextStep()(ctx)
    assert out["status"] == "skipped"
    assert out["skip_scoring"] is True


def test_load_model_artifacts_step_loads_each_enabled_model(monkeypatch):
    calls = {"loaded": []}

    class FakeSqlClient:
        def get_latest_model_artifact_details(self, *, model_key: str, trainer_key: str):
            calls["loaded"].append((model_key, trainer_key))
            return {"blob_container": "cont", "blob_path": f"models/{model_key}.pkl"}

    monkeypatch.setattr(pipes, "load_model_artifact", lambda details: f"artifact:{details['blob_path']}")

    ctx = {
        "sql_client": FakeSqlClient(),
        "model_specs": {"models": [{"model_key": "m1", "trainer_key": "t1"}, {"model_key": "m2", "trainer_key": "t2"}]},
    }
    out = pipes.LoadModelArtifactsStep()(ctx)
    assert calls["loaded"] == [("m1", "t1"), ("m2", "t2")]
    assert out["model_artifacts"]["m1"] == "artifact:models/m1.pkl"
    assert out["model_artifacts"]["m2"] == "artifact:models/m2.pkl"


def test_load_model_artifacts_step_skips_when_skip_scoring_flag_is_set():
    """Verify that artifact loading is skipped when scoring was skipped due to no data."""
    class FakeSqlClient:
        def get_latest_model_artifact_details(self, **kwargs):
            raise AssertionError("should not be called when skip_scoring is True")

    ctx = {
        "skip_scoring": True,
        "sql_client": FakeSqlClient(),
        "model_specs": {"models": [{"model_key": "m1", "trainer_key": "t1"}]},
    }
    out = pipes.LoadModelArtifactsStep()(ctx)
    assert out["skip_scoring"] is True
    assert "model_artifacts" not in out  # Not created when skipped


def test_score_models_step_scores_each_model_and_returns_context(monkeypatch):
    X = pd.DataFrame({"F": [1, 2]}, index=pd.Index([100, 200], name="ID"))

    def fake_score_model(*, model_object, X):
        return pd.Series([0.2, 0.8], index=X.index, name="score")

    monkeypatch.setattr(pipes, "score_model", fake_score_model)

    ctx = {"X_shared": X, "model_artifacts": {"m1": object(), "m2": object()}}
    out = pipes.ScoreModelsStep()(ctx)
    assert out["status"] == "scored"
    assert set(out["scoring_results"].keys()) == {"m1", "m2"}
    assert out["scoring_results"]["m1"].index.equals(X.index)


def test_score_models_step_skips_when_skip_scoring_flag_is_set():
    """Verify that scoring is skipped when skip_scoring flag is set."""
    ctx = {
        "skip_scoring": True,
        "X_shared": pd.DataFrame(),
        "model_artifacts": {"m1": object()},
    }
    out = pipes.ScoreModelsStep()(ctx)
    assert out["skip_scoring"] is True
    assert "scoring_results" not in out  # Not created when skipped


def test_combine_scoring_results_step_builds_wide_dataframe():
    base = pd.DataFrame({"ID": [1, 2], "Other": [9, 9]}).set_index(["ID"], drop=False)
    s1 = pd.Series([0.1, 0.2], index=base.index, name="score")
    s2 = pd.Series([0.3, 0.4], index=base.index, name="score")
    ctx = {
        "data": base,
        "resolved_entity_columns": ["ID"],
        "scoring_results": {"m1": s1, "m2": s2},
    }
    out = pipes.CombineScoringResultsStep()(ctx)
    df = out["scored_df"]
    assert list(df.columns) == ["ID", "m1", "m2"]
    assert df["m1"].tolist() == [0.1, 0.2]


def test_combine_scoring_results_step_skips_when_skip_scoring_flag_is_set():
    """Verify that result combining is skipped when skip_scoring flag is set."""
    ctx = {
        "skip_scoring": True,
        "data": pd.DataFrame({"ID": [1, 2]}),
        "resolved_entity_columns": ["ID"],
        "scoring_results": {"m1": pd.Series([0.1, 0.2])},
    }
    out = pipes.CombineScoringResultsStep()(ctx)
    assert out["status"] == "skipped"
    assert out["skip_scoring"] is True
    assert "scored_df" not in out  # Not created when skipped


def test_persist_scoring_results_step_adds_metadata_validates_and_writes(monkeypatch):
    captured = {}

    class FakeSqlClient:
        def truncate_table(self, *, table_name: str):
            captured["truncate_table"] = table_name

        def get_schema(self, *, table_name: str):
            captured["schema_table"] = table_name
            # Must match the real SqlClient.get_schema() contract, as the step builds a schema_dict
            # from these keys.
            return [
                {"column_name": "ID", "data_type": "int", "is_required": True},
                {"column_name": "m1", "data_type": "float", "is_required": False},
                {"column_name": "SystemEventId", "data_type": "nvarchar", "is_required": True},
                {"column_name": "ScoredAtUtc", "data_type": "datetimeoffset", "is_required": True},
                {"column_name": "ModelGroupKey", "data_type": "nvarchar", "is_required": True},
            ]

        def write_dataframe_to_table(self, *, df, table_name: str, if_exists: str = "append"):
            captured["df"] = df.copy()
            captured["table_name"] = table_name
            captured["if_exists"] = if_exists

    def fake_validate_transformed_data(*, transformed_data, target_schema):
        captured["validated"] = True
        captured["validated_df"] = transformed_data
        captured["validated_schema"] = target_schema

    monkeypatch.setattr(pipes, "validate_transformed_data", fake_validate_transformed_data)

    scored_df = pd.DataFrame({"ID": [1, 2], "m1": [0.1, 0.2]}).set_index(["ID"], drop=False)
    ctx = {
        "sql_client": FakeSqlClient(),
        "system_event_id": "sys",
        "model_group_key": "group",
        "model_specs": {"results_table_name": "dbo.Results"},
        "scored_df": scored_df,
    }

    out = pipes.PersistScoringResultsStep()(ctx)
    assert out["status"] == "persisted"
    # Verify truncate was called before writing
    assert captured["truncate_table"] == "dbo.Results"
    assert captured["schema_table"] == "dbo.Results"
    assert captured["table_name"] == "dbo.Results"
    assert captured["if_exists"] == "append"
    df = captured["df"]
    assert "SystemEventId" in df.columns
    assert "ScoredAtUtc" in df.columns
    assert "ModelGroupKey" in df.columns
    assert captured.get("validated") is True

    # validate_transformed_data receives the mutated scored_df + a richer schema dict
    assert captured["validated_df"] is ctx["scored_df"]
    assert captured["validated_schema"] == {
        "columns": ["ID", "m1", "SystemEventId", "ScoredAtUtc", "ModelGroupKey"],
        "data_types": {
            "ID": "int",
            "m1": "float",
            "SystemEventId": "nvarchar",
            "ScoredAtUtc": "datetimeoffset",
            "ModelGroupKey": "nvarchar",
        },
        "required": {
            "ID": True,
            "m1": False,
            "SystemEventId": True,
            "ScoredAtUtc": True,
            "ModelGroupKey": True,
        },
    }


def test_persist_scoring_results_step_skips_when_skip_scoring_flag_is_set():
    """Verify that table is still truncated even when skip_scoring flag is set."""
    truncate_called = {"called": False, "table": None}

    class FakeSqlClient:
        def truncate_table(self, *, table_name: str):
            truncate_called["called"] = True
            truncate_called["table"] = table_name

        def get_schema(self, **kwargs):
            raise AssertionError("get_schema should not be called when skip_scoring is True")

        def write_dataframe_to_table(self, **kwargs):
            raise AssertionError("write_dataframe_to_table should not be called when skip_scoring is True")

    ctx = {
        "skip_scoring": True,
        "sql_client": FakeSqlClient(),
        "system_event_id": "sys",
        "model_group_key": "group",
        "model_specs": {"results_table_name": "dbo.Results"},
        "scored_df": pd.DataFrame({"ID": [1, 2], "m1": [0.1, 0.2]}),
    }
    out = pipes.PersistScoringResultsStep()(ctx)
    assert out["status"] == "skipped"
    assert out["skip_scoring"] is True
    # Verify truncate was still called to clear stale predictions
    assert truncate_called["called"] is True
    assert truncate_called["table"] == "dbo.Results"

