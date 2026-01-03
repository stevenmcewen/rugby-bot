from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from functions.ml_models.helpers.trainer_factory import (
    parse_trainer_params,
    validate_xy,
    drop_na_rows,
    parse_time_or_none,
    sort_by_time,
    time_split,
    fit_model,
)


def test_parse_trainer_params_defaults_and_overrides():
    p = parse_trainer_params({})
    assert p.dropna is True
    assert p.val_fraction == 0.2
    assert p.use_early_stopping is True
    assert p.early_stopping_rounds == 50
    assert p.verbose is False

    p2 = parse_trainer_params({"dropna": False, "val_fraction": 0.1, "use_early_stopping": False, "early_stopping_rounds": 10, "verbose": True})
    assert p2.dropna is False
    assert p2.val_fraction == 0.1
    assert p2.use_early_stopping is False
    assert p2.early_stopping_rounds == 10
    assert p2.verbose is True


def test_validate_xy_raises_on_empty_or_mismatch():
    with pytest.raises(ValueError):
        validate_xy(pd.DataFrame(), pd.Series(dtype="int64"))

    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([1])
    with pytest.raises(ValueError, match="length mismatch"):
        validate_xy(X, y)


def test_drop_na_rows_keeps_alignment_for_weight_and_time():
    X = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, None])
    sw = pd.Series([0.1, 0.2, 0.3])
    t = pd.Series(["2025-01-01", "2025-01-02", "2025-01-03"])

    X2, y2, sw2, t2 = drop_na_rows(X, y, sw, t)
    assert len(X2) == len(y2) == len(sw2) == len(t2) == 1
    assert X2.iloc[0]["a"] == 1.0
    assert float(y2.iloc[0]) == 1.0
    assert float(sw2.iloc[0]) == 0.1
    assert t2.iloc[0] == "2025-01-01"


def test_parse_time_or_none_returns_none_when_all_unparseable():
    out = parse_time_or_none(pd.Series(["not-a-date", "also-bad"]))
    assert out is None


def test_sort_by_time_sorts_and_keeps_alignment():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    y = pd.Series([10, 20, 30], index=[0, 1, 2])
    t = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]), index=[0, 1, 2])
    sw = pd.Series([0.1, 0.2, 0.3], index=[0, 1, 2])

    X2, y2, sw2 = sort_by_time(X, y, t, sw)
    assert y2.tolist() == [20, 30, 10]
    assert sw2.tolist() == [0.2, 0.3, 0.1]


def test_time_split_splits_last_fraction_as_validation():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    sw = pd.Series([1, 1, 1, 1, 1])

    X_train, X_val, y_train, y_val, sw_train = time_split(X, y, sw, val_fraction=0.4)
    assert len(X_val) == 2
    assert X_val["a"].tolist() == [4, 5]
    assert y_val.tolist() == [4, 5]
    assert sw_train.tolist() == [1, 1, 1]


def test_fit_model_uses_early_stopping_when_validation_present():
    calls = {}

    class DummyModel:
        def fit(self, X_train, y_train, **kwargs):
            calls["kwargs"] = kwargs
            return self

    model = DummyModel()
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"a": [4]})
    y_val = pd.Series([1])

    out = fit_model(
        model,
        X_train=X_train,
        y_train=y_train,
        sample_weight_train=None,
        X_val=X_val,
        y_val=y_val,
        use_early_stopping=True,
        early_stopping_rounds=5,
        verbose=True,
    )

    assert out is model
    assert "eval_set" in calls["kwargs"]
    assert calls["kwargs"]["early_stopping_rounds"] == 5
    assert calls["kwargs"]["verbose"] is True


def test_fit_model_simple_fit_when_no_validation_or_disabled():
    calls = {"count": 0}

    class DummyModel:
        def fit(self, X_train, y_train, **kwargs):
            calls["count"] += 1
            # should not receive eval_set when not early stopping
            assert "eval_set" not in kwargs
            return self

    model = DummyModel()
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])

    out = fit_model(
        model,
        X_train=X_train,
        y_train=y_train,
        sample_weight_train=None,
        X_val=None,
        y_val=None,
        use_early_stopping=False,
        early_stopping_rounds=5,
        verbose=False,
    )

    assert out is model
    assert calls["count"] == 1


