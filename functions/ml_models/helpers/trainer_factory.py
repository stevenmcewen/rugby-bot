# ml/trainers/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd


# Trainer orchestration functions
# ============================================================

def train_xgb(
    *,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    time: Optional[pd.Series] = None,
    params: Optional[dict] = None,
) -> Any:
    """
    High-level orchestrator using small helpers.

    This is the function your trainers can call.
    """
    p = parse_trainer_params(params or {})

    validate_xy(X, y)

    if p.dropna:
        X, y, sample_weight, time = drop_na_rows(X, y, sample_weight, time)

    # No time split requested or available -> train on all data
    t = parse_time_or_none(time)
    if t is None or p.val_fraction <= 0:
        model.fit(X, y, sample_weight=sample_weight)
        return model

    # Time-based split
    X, y, sample_weight = sort_by_time(X, y, t, sample_weight)

    X_train, X_val, y_train, y_val, sw_train = time_split(
        X, y, sample_weight, p.val_fraction
    )

    return fit_model(
        model,
        X_train=X_train,
        y_train=y_train,
        sample_weight_train=sw_train,
        X_val=X_val,
        y_val=y_val,
        use_early_stopping=p.use_early_stopping,
        early_stopping_rounds=p.early_stopping_rounds,
        verbose=p.verbose,
    )

# Trainer steps helper functions
# ============================================================

@dataclass(frozen=True)
class TrainerParams:
    dropna: bool = True
    val_fraction: float = 0.2
    use_early_stopping: bool = True
    early_stopping_rounds: int = 50
    verbose: bool = False


def parse_trainer_params(params: dict) -> TrainerParams:
    """Read trainer params from a dict and apply defaults."""
    return TrainerParams(
        dropna=bool(params.get("dropna", True)),
        val_fraction=float(params.get("val_fraction", 0.2)),
        use_early_stopping=bool(params.get("use_early_stopping", True)),
        early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
        verbose=bool(params.get("verbose", False)),
    )


# Validation

def validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    """Basic shape checks so you fail fast and loudly."""
    if X is None or y is None:
        raise ValueError("X and y must be provided")
    if len(X) == 0:
        raise ValueError("X is empty")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)} len(y)={len(y)}")


# Cleaning

def drop_na_rows(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    time: Optional[pd.Series],
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    Drop rows where y is NA or any X feature is NA.
    Keeps sample_weight and time aligned.
    """
    mask = (~y.isna()) & (~X.isna().any(axis=1))
    X2 = X.loc[mask]
    y2 = y.loc[mask]
    sw2 = sample_weight.loc[mask] if sample_weight is not None else None
    t2 = time.loc[mask] if time is not None else None
    return X2, y2, sw2, t2


# Time parse

def parse_time_or_none(time: Optional[pd.Series]) -> Optional[pd.Series]:
    """Try parse time to datetime; if it totally fails return None."""
    if time is None:
        return None
    t = pd.to_datetime(time, errors="coerce")
    if t.isna().all():
        return None
    return t


def sort_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    sample_weight: Optional[pd.Series],
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Sort X/y/(sw) by ascending time index."""
    order = t.sort_values().index
    X2 = X.loc[order]
    y2 = y.loc[order]
    sw2 = sample_weight.loc[order] if sample_weight is not None else None
    return X2, y2, sw2


# SPLIT Train/Val

def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Split by row order (assumes already sorted by time).
    Validation = last val_fraction chunk.
    """
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    split_idx = max(1, n - n_val)

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    sw_train = sample_weight.iloc[:split_idx] if sample_weight is not None else None

    return X_train, X_val, y_train, y_val, sw_train


# Fit model

def fit_model(
    model: Any,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight_train: Optional[pd.Series],
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    use_early_stopping: bool,
    early_stopping_rounds: int,
    verbose: bool,
) -> Any:
    """
    Fit the model, optionally with early stopping if validation data exists.
    """
    if use_early_stopping and X_val is not None and len(X_val) > 0:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
    return model



# Trainer Classes
# ============================================================

class BaseTrainer:
    def train(
        self,
        *,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        time: Optional[pd.Series] = None,
        prediction_type: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError


class XgbClassifierV1Trainer(BaseTrainer):
    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or {}

    def train(
        self,
        *,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        time: Optional[pd.Series] = None,
        prediction_type: Optional[str] = None,
    ) -> Any:
        return train_xgb(
            model=model,
            X=X,
            y=y,
            sample_weight=sample_weight,
            time=time,
            params=self.params,
        )


class XgbRegressorV1Trainer(BaseTrainer):
    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or {}

    def train(
        self,
        *,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        time: Optional[pd.Series] = None,
        prediction_type: Optional[str] = None,
    ) -> Any:
        return train_xgb(
            model=model,
            X=X,
            y=y,
            sample_weight=sample_weight,
            time=time,
            params=self.params,
        )



# Factory Registration
# ============================================================

TrainerBuilder = Callable[[Optional[dict]], BaseTrainer]
TRAINER_REGISTRY: Dict[str, TrainerBuilder] = {}


def register_trainer(trainer_key: str) -> Callable[[TrainerBuilder], TrainerBuilder]:
    def decorator(builder: TrainerBuilder) -> TrainerBuilder:
        if trainer_key in TRAINER_REGISTRY:
            raise KeyError(f"Duplicate trainer_key registered: {trainer_key}")
        TRAINER_REGISTRY[trainer_key] = builder
        return builder
    return decorator


@register_trainer("xgb_classifier_v1")
def build_xgb_classifier_v1(params: Optional[dict] = None) -> BaseTrainer:
    return XgbClassifierV1Trainer(params=params)


@register_trainer("xgb_regressor_v1")
def build_xgb_regressor_v1(params: Optional[dict] = None) -> BaseTrainer:
    return XgbRegressorV1Trainer(params=params)


# FACTORY
# ============================================================

@dataclass(frozen=True)
class TrainerBuildContext:
    trainer_key: str
    trainer_params: Optional[dict] = None


class TrainerFactory:
    @staticmethod
    def create(ctx: TrainerBuildContext) -> BaseTrainer:
        if ctx.trainer_key not in TRAINER_REGISTRY:
            available = ", ".join(sorted(TRAINER_REGISTRY.keys()))
            raise KeyError(
                f"Unknown trainer_key='{ctx.trainer_key}'. Available: {available}"
            )

        params = ctx.trainer_params or {}
        if not isinstance(params, dict):
            raise ValueError("trainer_params must be a dictionary")

        builder = TRAINER_REGISTRY[ctx.trainer_key]
        return builder(params)



