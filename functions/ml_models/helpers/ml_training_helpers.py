from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from functions.logging.logger import get_logger
from functions.ml_models.ml_pipelines import TrainPayload

from functions.ml_models.helpers.model_factory import ModelFactory, ModelBuildContext
from functions.ml_models.helpers.trainer_factory import TrainerFactory, TrainerBuildContext

logger = get_logger(__name__)

# Training orchestration
def train_model(payload: TrainPayload) -> Any:
    """
    Train a model based on the provided training payload.

    Args:
        payload: TrainPayload containing all necessary training information.

    Returns:
        A trained (fitted) model instance.

    Raises:
        ValueError: If training fails.
    """
    try:
        # 1) Build the model using the ModelFactory
        logger.info(f"Building model with key: {payload.model_key}")
        model = ModelFactory.create(
            ModelBuildContext(
                model_key=payload.model_key,
                prediction_type=payload.prediction_type,
                model_params=payload.model_parameters,
            )
        )

        # 2) Build the trainer using the TrainerFactory (i.e. the parameters that the model will be trained with)
        logger.info(f"Building trainer with key: {payload.trainer_key}")
        trainer = TrainerFactory.create(
            TrainerBuildContext(
                trainer_key=payload.trainer_key,
                trainer_params=payload.trainer_parameters,
            )
        )
        # 3) Fit
        logger.info("Training model...")
        trained_model = trainer.train(
            model=model,
            X=payload.X,
            y=payload.y,
            sample_weight=payload.sample_weight,
            time=payload.time,
            prediction_type=payload.prediction_type,
        )

        return trained_model

    except Exception as e:
        logger.exception("Error training model")
        raise ValueError("Model training failed") from e

# Evaluation orchestration
def evaluate_model(
    *,
    model_object,
    prediction_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight=None,
    time=None,
) -> dict:
    """
    Evaluate a trained model using simple, sensible metrics.
    - classification: accuracy, F1, logloss, AUC
    - regression: RMSE, MAE, R2

    NOTE that this is a basic evaluation function for now, not factory pattern based. It can be extended later if needed."""

    # Drop rows where y is missing to avoid crashes
    mask = ~y.isna()
    # make sure X, y, sample_weight are still aligned after masking
    X_eval = X.loc[mask]
    y_eval = y.loc[mask]
    sw = sample_weight.loc[mask] if sample_weight is not None else None

    metrics = {}

    # CLASSIFICATION
    # ----------------------------------------------------------
    if prediction_type.lower() in {"classification", "binary"}:
        logger.info("Evaluating classification model")
        # raw predictions
        try:
            # Probability predictions
            y_prob = model_object.predict_proba(X_eval)[:, 1]
        except Exception:
            # Some XGB configs do not expose predict_proba, fall back to predict
            y_prob = model_object.predict(X_eval)
            y_prob = np.clip(y_prob, 0, 1)

        # class predictions, threshold at 0.5
        y_pred = (y_prob >= 0.5).astype(int)

        # metrics:
        # accuracy is % predicted correctly
        metrics["accuracy"] = accuracy_score(y_eval, y_pred, sample_weight=sw)
        # F1 score balances precision and recall i.e when we predict positive, how often are we correct, and how many actual positives did we catch
        metrics["f1"] = f1_score(y_eval, y_pred, sample_weight=sw)

        # logloss i.e. how well calibrated are the predicted probabilities 
        # e.g Predicting 0.51 for a win but being wrong is not punished heavily. Predicting 0.99 and being wrong is punished heavily.
        try:
            metrics["logloss"] = log_loss(y_eval, y_prob, sample_weight=sw)
        except ValueError:
            metrics["logloss"] = None

        # ROC AUC i.e. how well does the model separate the classes
        try:
            metrics["auc"] = roc_auc_score(y_eval, y_prob, sample_weight=sw)
        except ValueError:
            metrics["auc"] = None

        return metrics

    # REGRESSION
    # ----------------------------------------------------------
    elif prediction_type.lower() == "regression":
        logger.info("Evaluating regression model")
        y_pred = model_object.predict(X_eval)
        # metrics:
        # RMSE: root mean squared error, punishes large errors more heavily
        # e.g. predicting 10 when actual is 0 is worse than predicting 5 when actual is 0
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_eval, y_pred, sample_weight=sw)))
        # MAE: mean absolute error, average size of errors
        # e.g. predictions are on average 3.8 points off.
        metrics["mae"] = float(mean_absolute_error(y_eval, y_pred, sample_weight=sw))
        # R2: proportion of variance explained by the model
        # e.g. R2 of 0.65 means 65% of variance in target is explained by the model
        metrics["r2"] = float(r2_score(y_eval, y_pred, sample_weight=sw))

        return metrics

    # NOTE: Add other prediction types here as needed
    # ----------------------------------------------------------
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type!r}")
