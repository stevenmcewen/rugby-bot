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
    parsed_params = parse_trainer_params(params or {})

    # Validate the input data
    validate_xy(X, y)

    # Drop the rows where y is NA or any X feature is NA
    if parsed_params.dropna:
        X, y, sample_weight, time = drop_na_rows(X, y, sample_weight, time)

    # Parse the time or return None if it is not provided
    parsed_time = parse_time_or_none(time)
    if parsed_time is None or parsed_params.val_fraction <= 0:
        # Train on all data
        model.fit(X, y, sample_weight=sample_weight)
        return model

    # Sort the data by time
    X, y, sample_weight = sort_by_time(X, y, parsed_time, sample_weight)

    # Split the data into training and validation data
    X_train, X_val, y_train, y_val, sw_train = time_split(
        X, y, sample_weight, parsed_params.val_fraction
    )

    # Fit the model
    fitted_model = fit_model(
        model,
        X_train=X_train,
        y_train=y_train,
        sample_weight_train=sw_train,
        X_val=X_val,
        y_val=y_val,
        use_early_stopping=parsed_params.use_early_stopping,
        early_stopping_rounds=parsed_params.early_stopping_rounds,
        verbose=parsed_params.verbose,
    )

    return fitted_model

# Trainer steps helper functions
# ============================================================

@dataclass(frozen=True)
class TrainerParams:
    dropna: bool
    val_fraction: float
    use_early_stopping: bool
    early_stopping_rounds: int
    verbose: bool

# parse_trainer_params helper function
def parse_trainer_params(params: dict) -> TrainerParams:
    """
    This function read the trainer parameters from a dict supplied by the user and applies defaults if not provided.
    
    Accepts:
        params: dict
    Returns:
        TrainerParams
    """
    parsed_params = TrainerParams(
        dropna=bool(params.get("dropna", True)),
        val_fraction=float(params.get("val_fraction", 0.2)),
        use_early_stopping=bool(params.get("use_early_stopping", True)),
        early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
        verbose=bool(params.get("verbose", False)),
    )

    return parsed_params


# Validation
def validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    """This function validates the input data. It checks if the data is provided and if the length of the data is correct.
    Accepts:
        X: pd.DataFrame
        y: pd.Series
    Returns:
        None
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided")
    if len(X) == 0:
        raise ValueError("X is empty")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)} len(y)={len(y)}")

    return


# Cleaning
# drop_na_rows
def drop_na_rows(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    time: Optional[pd.Series],
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    This function drops the rows where y is NA or any X feature is NA.
    It keeps the sample_weight and time aligned.
    Accepts:
        X: pd.DataFrame
        y: pd.Series
        sample_weight: Optional[pd.Series]
        time: Optional[pd.Series]
    Returns:
        Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Optional[pd.Series]]
    """
    # Create a mask to identify rows where y is not NA and all X features are not NA
    mask = (~y.isna()) & (~X.isna().any(axis=1))
    # Apply the mask to the data to drop the rows where y is NA or any X feature is NA
    X2 = X.loc[mask]
    y2 = y.loc[mask]
    # Apply the mask to the sample_weight to keep the sample_weight aligned
    sw2 = sample_weight.loc[mask] if sample_weight is not None else None
    # Apply the mask to the time to keep the time aligned
    t2 = time.loc[mask] if time is not None else None
    # Return the cleaned data
    return X2, y2, sw2, t2


# Time parse
def parse_time_or_none(time: Optional[pd.Series]) -> Optional[pd.Series]:
    """This function parses the time to datetime. If it totally fails, it returns None.
    Accepts:
        time: Optional[pd.Series]
    Returns:
        Optional[pd.Series]
    """
    # Return None if time is not provided
    if time is None:
        return None
    # Convert the time to datetime
    time_dt = pd.to_datetime(time, errors="coerce")
    # Return None if the time is all NA
    if time_dt.isna().all():
        return None
    return time_dt


# sort_by_time
def sort_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    sample_weight: Optional[pd.Series],
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """This function sorts the data by time.
    Accepts:
        X: pd.DataFrame
        y: pd.Series
        t: pd.Series
        sample_weight: Optional[pd.Series]
    Returns:
        Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]
    """
    # Sort the data by time
    order = t.sort_values().index
    # Apply the order to the data
    X2 = X.loc[order]
    y2 = y.loc[order]
    # Apply the order to the sample_weight
    sw2 = sample_weight.loc[order] if sample_weight is not None else None
    # Return the sorted data
    return X2, y2, sw2


# SPLIT Train/Val
# time_split
def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]:
    """
    This function splits the data by row order (assumes already sorted by time).
    The validation data is the last val_fraction chunk.
    Accepts:
        X: pd.DataFrame
        y: pd.Series
        sample_weight: Optional[pd.Series]
        val_fraction: float
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]
    Validation = last val_fraction chunk.
    """
    # The total number of rows in the data
    n = len(X)
    # The number of rows in the validation data
    n_val = max(1, int(n * val_fraction))
    # The index to split the data
    split_idx = max(1, n - n_val)
    # Split the data into training and validation data
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    # Apply the order to the sample_weight
    sw_train = sample_weight.iloc[:split_idx] if sample_weight is not None else None
    # Return the split data
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
    Accepts:
        model: Any
        X_train: pd.DataFrame
        y_train: pd.Series
        sample_weight_train: Optional[pd.Series]
        X_val: Optional[pd.DataFrame]
        y_val: Optional[pd.Series]
        use_early_stopping: bool
        early_stopping_rounds: int
        verbose: bool
    Returns:
        Any
    """
    # Fit the model with early stopping if validation data exists
    if use_early_stopping and X_val is not None and len(X_val) > 0:
        # Fit the model with early stopping
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



