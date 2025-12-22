from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, Sequence, Any
from uuid import UUID

from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient
from functions.data_preprocessing.preprocessing_pipelines import run_preprocessing_pipeline


logger = get_logger(__name__)
settings = get_settings()

ModelContext = dict[str, Any]

## Pipeline step interface ##
class ModelStep(Protocol):
    """
    Single responsibility step in the model pipeline.

    each step:
    - accepts a mutable `ModelContext`
    - performs exactly one responsibility
    - returns the updated context
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        ...


class ModelPipeline:
    """
    Light‑weight orchestration of a sequence of `ModelStep`s.
    The pipeline is responsible for:
    - Running each step in sequence.
    - Passing an `ModelContext` between steps.
    - Logging step execution.
    """

    def __init__(self, *steps: ModelStep) -> None:
        self._steps: tuple[ModelStep, ...] = steps

    def run(self, initial_context: ModelContext) -> ModelContext:
        context = initial_context
        for step in self._steps:
            step_name = step.__class__.__name__
            logger.info("Running model step: %s", step_name)
            context = step(context)
        return context


### Step implementations: Shared pipelines ###

class LoadModelSpecStep:
    """
    This step loads the model specifications for the given model_group_key. And adds it to the context.
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Loading model specification for model_group_key: %s", context["model_group_key"])

        try:
            sql_client: SqlClient = context["sql_client"]
            model_group_key: str = context["model_group_key"]
            model_specs = sql_client.get_model_specs_by_model_group_key(model_group_key=model_group_key)
            context["model_specs"] = model_specs
        except Exception as e:
            logger.error("Error getting model_specs by model_group_key: %s", e)
            raise
        return context
    
class LoadDataStep:
    """
    This step loads the training data for the given model_group_key. And adds it to the context.
    """
    
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Loading training data for model_group_key: %s", context["model_group_key"])
        try:
            # get the sql client and the model specifications
            sql_client: SqlClient = context["sql_client"]
            specs: dict = context["model_specs"]
            run_type: str = context["run_type"]
            if run_type == "training":
                dataset_source = specs["training_dataset_source"]
            elif run_type == "scoring":
                dataset_source = specs["scoring_dataset_source"]
            else:
                raise ValueError(f"Invalid run type: {run_type}")
            feature_columns: list[str] = specs["columns"]["feature"]
            target_column: str = specs["columns"]["target"]
            sample_weight_column: str = specs["columns"]["weight"]
            time_column: str = specs["columns"]["time"]
            # create a list of the columns to select
            columns_to_select: list[str] = feature_columns + [target_column] + [sample_weight_column] + [time_column]
            # get the data from the database
            data = sql_client.get_model_source_data(
                source_table=dataset_source,
                columns_to_select=columns_to_select)
            # add the data to the context
            context["data"] = data
            logger.info("Data loaded successfully for model_group_key: %s", context["model_group_key"])

        except Exception as e:
            logger.error("Error loading data for model_group_key: %s", e)
            raise
        return context

### Step implementations: Training pipelines ###
class StartTrainingRunStep:
    """
    This step starts the training run. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Starting training run")
        context["status"] = "started"
        context["run_type"] = "training"
        return context

class TrainModelStep:
    """
    This step trains the model. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Training model")
        context["status"] = "training"
        # insert logic to call the training helpers here
        return context

class EvaluateModelStep:
    """
    This step evaluates the model. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Evaluating model")
        context["status"] = "evaluating"
        # insert logic to call the evaluation helpers here
        return context

class PersistModelArtifactsStep:
    """
    This step persists the model artifacts. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Persisting model artifacts")
        context["status"] = "persisting"
        # insert logic to call the persistence helpers here
        return context

class FinalizeTrainingRunStep:
    """
    This step finalizes the training run. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Finalizing training run")
        context["status"] = "completed"
        # insert logic to call the finalization helpers here
        return context


### Step implementations: Scoring pipelines ###
class StartScoringRunStep:
    """
    This step starts the training run. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Starting training run")
        context["status"] = "started"
        context["run_type"] = "scoring"
        return context

class LoadModelArtifactsStep:

class ScoreModelStep:

class PersistScoringResultsStep:

class FinalizeScoringRunStep:

### Pipeline registry / factory ###
ModelPipelineFactory = Callable[[], Sequence[ModelStep]]
MODEL_PIPELINE_REGISTRY: dict[str, ModelPipelineFactory] = {}


# factory registry for the model pipelines
def register_model_pipeline(name: str, factory: ModelPipelineFactory) -> None:
    """
    Register a named orchestration pipeline for the model layer build.

    """
    if name in MODEL_PIPELINE_REGISTRY:
        raise ValueError(f"Model pipeline {name!r} is already registered.")
    MODEL_PIPELINE_REGISTRY[name] = factory

# factory builder for the model pipelines
def build_model_pipeline_for(name: str) -> ModelPipeline:
    """
    Resolve a registered orchestration pipeline instance.
    """
    try:
        factory = MODEL_PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"No model pipeline registered under name {name!r}") from exc

    steps = factory()
    return ModelPipeline(*steps)

#### pipeline factories ###
# NOTE: Add more pipelines here if you want to add more steps to the pipeline
### factory for the default model training pipeline ###
def default_model_training_pipeline_factory() -> tuple[ModelStep, ...]:
    """
    Default end‑to‑end orchestration for model training:
        1. Start the training run
        2. Load the model specification
        3. Load the training data
        4. Train the model
        5. Evaluate the model
        6. Persist the model artifacts
        7. Finalize the training run
        """

    pipeline_steps = (
        StartTrainingRunStep(),
        LoadModelSpecStep(),
        LoadDataStep(),
        TrainModelStep(),
        EvaluateModelStep(),
        PersistModelArtifactsStep(),
        FinalizeTrainingRunStep(),
    )
    return pipeline_steps

### factory for the default model scoring pipeline ###
def default_model_scoring_pipeline_factory() -> tuple[ModelStep, ...]:
    """
    Default end‑to‑end orchestration for model scoring:
        1. Start the scoring run
        2. Load the model specification
        3. Load the scoring data
        4. Load the model artifacts
        5. Score the model
        6. Persist the scoring results
        7. Finalize the scoring run
        """

    pipeline_steps = (
        StartScoringRunStep(),
        LoadModelSpecStep(),
        LoadDataStep(),
        LoadModelArtifactsStep(),
        ScoreModelStep(),
        PersistScoringResultsStep(),
        FinalizeScoringRunStep(),
    )
    return pipeline_steps

# Register the default orchestration pipeline at import time.
#NOTE: Register the pipelines here if you want to use them in the orchestrator
register_model_pipeline("default_model_training", default_model_training_pipeline_factory)
register_model_pipeline("default_model_scoring", default_model_scoring_pipeline_factory)