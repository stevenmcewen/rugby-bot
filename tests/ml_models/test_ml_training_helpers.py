from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from functions.ml_models.helpers import ml_training_helpers as helpers


def test_train_model_calls_model_and_trainer_factories(monkeypatch):
    created = {}

    class DummyTrainer:
        def train(self, *, model, X, y, sample_weight=None, time=None, prediction_type=None):
            created["trained_with"] = {
                "model": model,
                "X_shape": X.shape,
                "y_len": len(y),
                "sample_weight_is_none": sample_weight is None,
                "time_is_none": time is None,
                "prediction_type": prediction_type,
            }
            return "trained-model"

    monkeypatch.setattr(helpers, "ModelFactory", SimpleNamespace(create=lambda _ctx: "model-obj"))
    monkeypatch.setattr(helpers, "TrainerFactory", SimpleNamespace(create=lambda _ctx: DummyTrainer()))

    payload = SimpleNamespace(
        model_key="international_rugby_homewin_v1",
        trainer_key="xgb_classifier_v1",
        prediction_type="classification",
        X=pd.DataFrame({"a": [1, 2, 3]}),
        y=pd.Series([0, 1, 0]),
        sample_weight=None,
        time=None,
        model_parameters={},
        trainer_parameters={},
    )

    out = helpers.train_model(payload)
    assert out == "trained-model"
    assert created["trained_with"]["prediction_type"] == "classification"


def test_evaluate_model_classification_uses_predict_proba_when_available():
    class DummyModel:
        def predict_proba(self, X):
            # Alternate probabilities so predictions match y=[1,0,1,0]
            # Row0 -> class1, Row1 -> class0, ...
            probs = []
            for i in range(len(X)):
                if i % 2 == 0:
                    probs.append([0.1, 0.9])
                else:
                    probs.append([0.9, 0.1])
            return np.array(probs)

    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    # include both classes so logloss/auc are defined
    y = pd.Series([1, 0, 1, 0])

    metrics = helpers.evaluate_model(
        model_object=DummyModel(),
        prediction_type="classification",
        X=X,
        y=y,
        sample_weight=None,
        time=None,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["logloss"] is not None
    assert metrics["auc"] is not None


def test_evaluate_model_classification_falls_back_to_predict_when_no_predict_proba():
    class DummyModel:
        def predict(self, X):
            return np.array([0.9] * len(X))

    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 1, 1])

    metrics = helpers.evaluate_model(
        model_object=DummyModel(),
        prediction_type="binary",
        X=X,
        y=y,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_model_regression_returns_rmse_mae_r2():
    class DummyModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0])

    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 4.0])

    metrics = helpers.evaluate_model(
        model_object=DummyModel(),
        prediction_type="regression",
        X=X,
        y=y,
    )

    assert metrics["rmse"] == pytest.approx(np.sqrt(((0) ** 2 + (0) ** 2 + (-1) ** 2) / 3))
    assert metrics["mae"] == pytest.approx((0 + 0 + 1) / 3)
    assert "r2" in metrics



