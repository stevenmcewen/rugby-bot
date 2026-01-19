from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def score_model(model_object: Any, X: pd.DataFrame) -> pd.Series:
    """
    Score a trained model using the provided feature matrix X.

    Accepts:
        model_object: The trained model instance to use for scoring.
        X: A pandas DataFrame containing the feature matrix for scoring.
    
    Returns:
        A pandas Series containing the model scores that is aligned with the input DataFrame X.

    Contract:
    - Returns a pandas Series with the SAME index as X (for safe alignment / joins).
    - For classifiers, returns positive-class probability when available.
    - For regressors, returns the raw numeric prediction.
    """
    if model_object is None:
        raise ValueError("score_model: model_object is required")
    if X is None or not isinstance(X, pd.DataFrame):
        raise TypeError("score_model: X must be a pandas DataFrame")

    # Convert string columns to categorical to be compatible with XGBoost
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype == "string":
            X[col] = X[col].astype("category")

    ## Predict Probabilities ##
    if hasattr(model_object, "predict_proba"):
        probabilities = model_object.predict_proba(X)
        # Ensure output is a numpy array for shape validation.
        probabilities = np.asarray(probabilities)
        # Validate shape of predict_proba output, output must be a 2D array and number of rows must match X (input rows).
        if probabilities.ndim != 2 or probabilities.shape[0] != len(X):
            raise ValueError(
                f"score_model: predict_proba returned unexpected shape {probabilities.shape} for X rows={len(X)}"
            )
        # For binary classification, we expect 2 columns (negative class, positive class).
        if probabilities.shape[1] < 2:
            raise ValueError(
                f"score_model: predict_proba returned {probabilities.shape[1]} columns; expected >= 2 for binary classification"
            )
        # Extract positive class probabilities and return as a Series aligned to X's index.
        scores = probabilities[:, 1]
        score_series = pd.Series(scores, index=X.index, name="score", dtype="float64")
        return score_series

    ## Predict Values ##
    if not hasattr(model_object, "predict"):
        raise TypeError("score_model: model_object has neither predict_proba nor predict")

    predictions = model_object.predict(X)
    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or predictions.shape[0] != len(X):
        # Some models may return (n,1); flatten it safely.
        predictions = predictions.reshape(-1)
    if predictions.shape[0] != len(X):
        raise ValueError(
            f"score_model: predict returned unexpected shape {predictions.shape} for X rows={len(X)}"
        )

    score_series = pd.Series(predictions.astype("float64"), index=X.index, name="score")
    return score_series