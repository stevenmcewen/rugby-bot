from __future__ import annotations

from typing import Any

from logging.logger import getLogger
from functions.sql.sql_client import SqlClient
from functions.ml_models.ml_pipelines import TrainPayload

from functions.ml_models.helpers.model_factory import ModelFactory, ModelBuildContext
from functions.ml_models.helpers.trainer_factory import TrainerFactory, TrainerBuildContext

logger = getLogger(__name__)


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

