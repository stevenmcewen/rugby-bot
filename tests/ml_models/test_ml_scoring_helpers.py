from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from functions.ml_models.helpers.ml_scoring_helpers import score_model


class DummyClassifier:
    def predict_proba(self, X: pd.DataFrame):
        # Return 2-class probabilities
        n = len(X)
        p1 = np.linspace(0.1, 0.9, n)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


class DummyRegressor:
    def predict(self, X: pd.DataFrame):
        return np.arange(len(X), dtype=float) * 3.5


def test_score_model_classifier_returns_series_aligned_to_index():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([101, 102, 103], name="ID"))
    s = score_model(DummyClassifier(), X)
    assert isinstance(s, pd.Series)
    assert s.index.equals(X.index)
    assert s.dtype == "float64"
    assert np.isclose(s.iloc[0], 0.1)
    assert np.isclose(s.iloc[-1], 0.9)


def test_score_model_regressor_returns_series_aligned_to_index():
    X = pd.DataFrame({"a": [1, 2]}, index=pd.Index([7, 8], name="ID"))
    s = score_model(DummyRegressor(), X)
    assert isinstance(s, pd.Series)
    assert s.index.equals(X.index)
    assert list(s.values) == [0.0, 3.5]


def test_score_model_raises_on_missing_model():
    X = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="model_object is required"):
        score_model(None, X)


def test_score_model_raises_on_non_dataframe_X():
    with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
        score_model(DummyClassifier(), X=None)  # type: ignore[arg-type]


def test_score_model_predict_proba_validates_shape_rows(monkeypatch):
    class BadClassifier:
        def predict_proba(self, X):
            # wrong row count
            return np.zeros((len(X) + 1, 2))

    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="unexpected shape"):
        score_model(BadClassifier(), X)


def test_score_model_predict_proba_requires_two_columns():
    class OneColClassifier:
        def predict_proba(self, X):
            return np.zeros((len(X), 1))

    X = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="expected >= 2"):
        score_model(OneColClassifier(), X)


def test_score_model_raises_when_no_predict_or_predict_proba():
    class NoPredict:
        pass

    X = pd.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="neither predict_proba nor predict"):
        score_model(NoPredict(), X)


def test_score_model_predict_flattens_n_by_1_predictions():
    class NBy1Regressor:
        def predict(self, X):
            return np.arange(len(X), dtype=float).reshape(-1, 1)

    X = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 11, 12], name="ID"))
    s = score_model(NBy1Regressor(), X)
    assert s.index.equals(X.index)
    assert s.tolist() == [0.0, 1.0, 2.0]

