from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence
from datetime import datetime, timezone

from functions.config.settings import get_settings
from functions.data_preprocessing.preprocessing_helpers import validate_transformed_data
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient
from functions.utils.utils import stable_schema_hash, uct_now_iso

from functions.ml_models.helpers.ml_training_helpers import train_model, evaluate_model
from functions.ml_models.helpers.ml_utils import serialize_model_artifact, persist_model_artifact, load_model_artifact
from functions.ml_models.helpers.ml_scoring_helpers import score_model



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

# data classes for the training and scoring payloads
@dataclass
class TrainPayload:
    model_key: str
    trainer_key: str
    prediction_type: str

    entity_columns: List[str]
    feature_columns: List[str]
    target_column: str

    sample_weight_column: Optional[str]
    time_column: Optional[str]

    X: pd.DataFrame
    y: pd.Series
    sample_weight: Optional[pd.Series]
    time: Optional[pd.Series]
    model_parameters: Dict[str, Any]
    trainer_parameters: Dict[str, Any]


@dataclass
class TrainedModelResult:
    """
    Output contract from training.
    Keep it consistent so Evaluate + Persist steps are boring.
    """
    model_key: str
    trainer_key: str
    prediction_type: str

    target_column: str
    feature_columns: List[str]
    entity_columns: List[str]
    schema_hash: str
    trained_at_utc: str

    model_object: Any
    train_rows: int


@dataclass
class EvaluationResult:
    model_key: str
    metrics: Dict[str, float]
    evaluated_at_utc: str

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
    Load training or scoring data for the given model_group_key, based on model spec.
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Loading data for model_group_key: %s", context["model_group_key"])

        try:
            sql_client: SqlClient = context["sql_client"]
            model_specs: dict = context["model_specs"]
            run_type: str = context["run_type"]

            if run_type == "training":
                dataset_source = model_specs["training_dataset_source"]
            elif run_type == "scoring":
                dataset_source = model_specs["scoring_dataset_source"]
            else:
                raise ValueError(f"Invalid run type: {run_type!r}")

            entity_cols: list[str] = model_specs["columns"].get("entity", []) or []
            feature_cols: list[str] = model_specs["columns"].get("feature", []) or []

            models: list[dict] = model_specs.get("models", []) or []

            # Targets should be selected ONLY for training
            target_cols: list[str] = []
            if run_type == "training":
                target_cols = [m["target_column"] for m in models if m.get("target_column")]

            # Weight column: prefer group-level, allow per-model overrides
            weight_cols: list[str] = []
            group_weight = model_specs["columns"].get("weight")
            if group_weight:
                weight_cols.append(group_weight)

            # If there is a per-model sample_weight_column, include those too (rare but allowed)
            weight_cols.extend([model["sample_weight_column"] for model in models if model.get("sample_weight_column")])

            # Time columns: useful for splits/validation; usually present in both training + scoring views
            time_cols: list[str] = [model["time_column"] for model in models if model.get("time_column")]

            # Deduplicate while preserving order
            columns_to_select: list[str] = list(
                dict.fromkeys(entity_cols + feature_cols + target_cols + weight_cols + time_cols)
            )

            data = sql_client.get_model_source_data(
                source_table=dataset_source,
                columns_to_select=columns_to_select,
            )

            context["data"] = data
            context["columns_selected"] = columns_to_select
            logger.info("Data loaded successfully from %s", dataset_source)

        except Exception as e:
            logger.error("Error loading data for model_group_key='%s': %s", context.get("model_group_key"), e)
            raise
        return context

### Step implementations: Training pipelines ###
class StartTrainingRunStep:
    """
    Mark context as a training run.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Starting training run")
        context["status"] = "started"
        context["run_type"] = "training"
        return context
    
class PrepareTrainingContextStep:
    """
    Prepare training context:
      - validates presence of model_spec + data
      - validates the target columns exist in the data
      - validates the feature columns exist in the data
      - resolves the shared columns
      - resolves the weight column fallback from group-level spec
      - pre-computes the schema hash
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Preparing training context")
        context["status"] = "preparing_training"

        data: pd.DataFrame = context.get("data")
        spec: dict = context.get("model_specs")

        # validate the data is a pandas DataFrame and it exists
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("PrepareTrainingContextStep expected context['data'] as a pandas DataFrame")

        if not spec or not isinstance(spec, dict):
            raise ValueError("PrepareTrainingContextStep expected context['model_specs'] as a dict spec bundle")

        columns = spec.get("columns") or {}
        feature_columns: List[str] = columns.get("feature", []) or []
        entity_columns: List[str] = columns.get("entity", []) or []
        group_weight_column: Optional[str] = columns.get("weight")

        # Set a stable index for alignment of multiple models to the same entity identifiers.
        # Keeping drop=False ensures the entity values remain available as columns too.
        if entity_columns:
            data = data.copy()
            data = data.set_index(entity_columns, drop=False)
            context["data"] = data

        # get list of models
        models: List[dict] = spec.get("models", []) or []
        # validate the target columns exist in the data
        targets = [m.get("target_column") for m in models if m.get("target_column")]
        # validate that the target columns exist in the data
        missing_targets = [t for t in targets if t not in data.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns in training data: {missing_targets}")

        # validate the feature columns exist in the data
        if not feature_columns:
            raise ValueError("No feature columns found in model_specs['columns']['feature']")

        # validate that the feature columns exist in the data
        missing_features = [column for column in feature_columns if column not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in data: {missing_features}")

        # resolve the shared columns
        context["resolved_feature_columns"] = feature_columns
        context["resolved_entity_columns"] = entity_columns
        context["resolved_group_weight_column"] = group_weight_column
        # pre-compute the schema hash which is used to validate the features used in the model training 
        # This is used to validate that identical features are used in the model training and scoring
        context["resolved_schema_hash"] = stable_schema_hash(feature_columns)

        # Keep X once for all models (saves time/memory)
        context["X_shared"] = data[feature_columns].copy()

        return context

class BuildTrainingPayloadsStep:
    """
    Build TrainPayload objects for each enabled model.
    Stores: context['train_payloads'] = {model_key: TrainPayload}
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Building training payloads")
        context["status"] = "building_train_payloads"

        # get the data and the model specs from the context
        data: pd.DataFrame = context["data"]
        spec: dict = context["model_specs"]

        feature_columns: List[str] = context["resolved_feature_columns"]
        entity_columns: List[str] = context["resolved_entity_columns"]
        group_weight_column: Optional[str] = context.get("resolved_group_weight_column")

        # get the list of models from the model specs
        models: List[dict] = spec.get("models", []) or []
        if not models:
            raise ValueError("No enabled models found in model_specs['models']")

        # get the shared features from the context
        X_shared: pd.DataFrame = context.get("X_shared")
        # if the shared features are not in the context, create them
        if X_shared is None:
            X_shared = data[feature_columns].copy()

        # create a dictionary to store the train payloads
        train_payloads: Dict[str, TrainPayload] = {}

        # iterate over each model and build the train payloads for each model
        for m in models:
            model_key = m["model_key"]
            trainer_key = m["trainer_key"]
            target_column = m["target_column"]
            prediction_type = m["prediction_type"]
            model_parameters = m.get("model_parameters") or {}
            trainer_parameters = m.get("trainer_parameters") or {}

            # Weight/time: per-model override > group-level weight role
            sample_weight_column = m.get("sample_weight_column") or group_weight_column
            time_column = m.get("time_column")

            # Validate target exists (training only)
            if target_column not in data.columns:
                raise ValueError(f"Missing target column {target_column!r} for model_key={model_key!r}")

            y = data[target_column].copy()

            # initialize the sample weight to None
            sample_weight = None
            # if the sample weight column is in the data, get the sample weight
            if sample_weight_column:
                if sample_weight_column not in data.columns:
                    raise ValueError(
                        f"Missing weight column {sample_weight_column!r} for model_key={model_key!r}"
                    )
                else:
                    sample_weight = data[sample_weight_column].copy()

            # initialize the time values to None
            time_values = None
            # if the time column is in the data, get the time values
            if time_column:
                if time_column not in data.columns:
                    logger.warning(
                        "Time column %s not found for model_key=%s; time-based split may be skipped",
                        time_column, model_key
                    )
                else:
                    time_values = data[time_column].copy()

            # create the train payload for the model
            train_payloads[model_key] = TrainPayload(
                model_key=model_key,
                trainer_key=trainer_key,
                prediction_type=prediction_type,
                entity_columns=entity_columns,
                feature_columns=feature_columns,
                target_column=target_column,
                sample_weight_column=sample_weight_column,
                time_column=time_column,
                X=X_shared,
                y=y,
                sample_weight=sample_weight,
                time=time_values,
                model_parameters=model_parameters,
                trainer_parameters=trainer_parameters,
            )

            # log the train payload for the model
            logger.info(
                "Prepared TrainPayload model_key=%s trainer_key=%s target=%s type=%s",
                model_key, trainer_key, target_column, prediction_type
            )

        context["train_payloads"] = train_payloads
        return context

class TrainModelsStep:
    """
    Train each model from payloads.
    Stores: context['trained_models'] = {model_key: TrainedModelResult}
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Training models")
        context["status"] = "training_models"

        # get the schema hash and the train payloads from the context
        schema_hash: str = context["resolved_schema_hash"]
        train_payloads: Dict[str, TrainPayload] = context.get("train_payloads") or {}
        # validate that the train payloads exist
        if not train_payloads:
            raise ValueError("No train_payloads found in context. Run BuildTrainingPayloadsStep first.")

        # create a dictionary to store the trained models
        trained_models: Dict[str, TrainedModelResult] = {}

        # iterate over each train payload and train the model
        for model_key, payload in train_payloads.items():
            # log the model key and the trainer key
            logger.info("Training model_key=%s with trainer_key=%s", model_key, payload.trainer_key)

            try:
                # train the model
                model_obj = train_model(payload)
                # create the trained model result
                trained_models[model_key] = TrainedModelResult(
                    model_key=payload.model_key,
                    trainer_key=payload.trainer_key,
                    prediction_type=payload.prediction_type,
                    target_column=payload.target_column,
                    feature_columns=payload.feature_columns,
                    entity_columns=payload.entity_columns,
                    schema_hash=schema_hash,
                    trained_at_utc=uct_now_iso(),
                    model_object=model_obj,
                    train_rows=int(payload.X.shape[0]),
                )
            except Exception as e:
                logger.error("Error training model_key=%s: %s", model_key, e)
                raise

        # add the trained models to the context
        context["trained_models"] = trained_models
        context["status"] = "trained"
        return context

class EvaluateModelsStep:
    """
    Evaluate each trained model.
    Stores: context['evaluation_results'] = {model_key: EvaluationResult}
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Evaluating models")
        context["status"] = "evaluating_models"

        # get the trained models and the train payloads from the context
        trained_models: Dict[str, TrainedModelResult] = context.get("trained_models") or {}
        if not trained_models:
            raise ValueError("No trained_models found in context. Run TrainModelsStep first.")

        train_payloads: Dict[str, TrainPayload] = context.get("train_payloads") or {}
        if not train_payloads:
            raise ValueError("No train_payloads found in context. Run BuildTrainingPayloadsStep first.")

        evaluation_results: Dict[str, EvaluationResult] = {}

        # iterate over each trained model and evaluate the model
        for model_key, tm in trained_models.items():
            logger.info("Evaluating model_key=%s", model_key)

            payload = train_payloads.get(model_key)
            if payload is None:
                raise ValueError(f"Missing TrainPayload for model_key={model_key!r}")

            try:
                # evaluate the model
                metrics = evaluate_model(
                    model_object=tm.model_object,
                    prediction_type=tm.prediction_type,
                    X=payload.X,
                    y=payload.y,
                        sample_weight=payload.sample_weight,
                        time=payload.time,
                    )
            except Exception as e:
                logger.error("Error evaluating model_key=%s: %s", model_key, e)
                raise

            # create the evaluation result
            evaluation_results[model_key] = EvaluationResult(
                model_key=model_key,
                metrics=metrics,
                evaluated_at_utc=uct_now_iso(),
            )

        # add the evaluation results to the context
        context["evaluation_results"] = evaluation_results
        context["status"] = "evaluated"
        return context


class PersistModelArtifactsStep:
    """
    Persist each trained model artifact + its evaluation metrics.
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Persisting model artifacts")
        context["status"] = "persisting_artifacts"

        sql_client: "SqlClient" = context["sql_client"]
        system_event_id = context["system_event_id"]

        trained_models: Dict[str, TrainedModelResult] = context.get("trained_models") or {}
        eval_results: Dict[str, EvaluationResult] = context.get("evaluation_results") or {}

        if not trained_models:
            raise ValueError("No trained_models found in context. Run TrainModelsStep first.")

        for model_key, tm in trained_models.items():
            metrics = eval_results.get(model_key).metrics if model_key in eval_results else {}
            try:
                logger.info("Persisting artifact for model_key=%s", model_key)
                artifact_bytes = serialize_model_artifact(tm.model_object)

                artefact_id, artefact_version = persist_model_artifact(
                    sql_client=sql_client,
                    system_event_id=system_event_id,
                    model_key=tm.model_key,
                    trainer_key=tm.trainer_key,
                    prediction_type=tm.prediction_type,
                    target_column=tm.target_column,
                    schema_hash=tm.schema_hash,
                    metrics=metrics,
                    artifact_bytes=artifact_bytes,
                )

                logger.info(
                    "Persisted model_key=%s artifact_id=%s version=%s",
                    model_key, artefact_id, artefact_version
                )
            except Exception as e:
                logger.error("Error persisting artifact for model_key=%s: %s", model_key, e)
                raise

        context["status"] = "persisted"
        return context

### Step implementations: Scoring pipelines ###

class StartScoringRunStep:
    """
    This step starts the training run. And adds it to the context.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Starting scoring run")
        context["status"] = "started"
        context["run_type"] = "scoring"
        return context

class PrepareScoringContextStep:
    """
    Prepare scoring context:
      - validates presence of model_spec + data
      - resolves shared feature columns
      - pre-computes schema hash (feature-only)
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Preparing scoring context")
        context["status"] = "preparing_scoring"

        data: pd.DataFrame = context.get("data")
        spec: dict = context.get("model_specs")

        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("PrepareScoringContextStep expected context['data'] as a pandas DataFrame")
        if not spec or not isinstance(spec, dict):
            raise ValueError("PrepareScoringContextStep expected context['model_specs'] as a dict spec bundle")

        columns = spec.get("columns") or {}
        feature_columns: List[str] = columns.get("feature", []) or []
        entity_columns: List[str] = columns.get("entity", []) or []

        # validate the feature columns exist in the data
        if not feature_columns:
            raise ValueError("No feature columns found in model_specs['columns']['feature']")

        # Validate entity cols exist in our data (this is needed for aligning scores from multiple models back to entities)
        missing_entities = [c for c in entity_columns if c not in data.columns]
        if missing_entities:
            raise ValueError(f"Missing entity columns in scoring data: {missing_entities}")

        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in scoring data: {missing_features}")

        context["resolved_feature_columns"] = feature_columns
        context["resolved_entity_columns"] = entity_columns
        context["resolved_schema_hash"] = stable_schema_hash(feature_columns)

        # Set a stable index for alignment of multiple models to the same entity identifiers.
        # Keeping drop=False ensures the entity values remain available as columns too.
        if entity_columns:
            data = data.copy()
            data = data.set_index(entity_columns, drop=False)
            context["data"] = data

        context["X_shared"] = data[feature_columns].copy()
        return context


class LoadModelArtifactsStep:
    """
    Load model artifacts from for each model in the spec.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Loading model artifacts")

        for model_spec in context["model_specs"].get("models", []):
            model_key = model_spec["model_key"]
            trainer_key = model_spec["trainer_key"]

            logger.info("Loading artifact for model_key=%s trainer_key=%s", model_key, trainer_key)
            try:
                sql_client: SqlClient = context["sql_client"]
                artifact_details = sql_client.get_latest_model_artifact_details(
                    model_key=model_key,
                    trainer_key=trainer_key,
                )
                if artifact_details is None:
                    raise ValueError(f"No artifact found for model_key={model_key!r} trainer_key={trainer_key!r}")
                
                artifact = load_model_artifact(artifact_details)

                if artifact is None:
                    raise ValueError(f"Failed to load artifact from path: {artifact_details['blob_path']!r}")

                if "model_artifacts" not in context:
                    context["model_artifacts"] = {}
                context["model_artifacts"][model_key] = artifact
            except Exception as e:
                logger.error("Error loading artifact for model_key=%s trainer_key=%s: %s", model_key, trainer_key, e)
                raise
        return context


class ScoreModelsStep:
    """
    Score each loaded model against the shared features.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Scoring models")

        X_shared: pd.DataFrame = context.get("X_shared")
        if X_shared is None:
            raise ValueError("No shared features found in context. Run PrepareScoringContextStep first.")
        
        model_artifacts: Dict[str, Any] = context.get("model_artifacts") or {}
        if not model_artifacts:
            raise ValueError("No model_artifacts found in context. Run LoadModelArtifactsStep first.")
        
        scoring_results: Dict[str, pd.Series] = {}
        for model_key, model_obj in model_artifacts.items():
            logger.info("Scoring model_key=%s", model_key)
            try:
                series_scores = score_model(
                    model_object=model_obj,
                    X=X_shared,
                )
                scoring_results[model_key] = series_scores
            except Exception as e:
                logger.error("Error scoring model_key=%s: %s", model_key, e)
                raise
        context["scoring_results"] = scoring_results
        context["status"] = "scored"
        return context


class CombineScoringResultsStep:
    """
    Combine entity columns + per-model scoring series into one wide DataFrame.
    """

    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Combining scoring results")

        data: pd.DataFrame = context.get("data")
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("CombineScoringResultsStep expected context['data'] as a pandas DataFrame")

        entity_cols: List[str] = context.get("resolved_entity_columns") or []
        scoring_results: Dict[str, pd.Series] = context.get("scoring_results") or {}
        if not scoring_results:
            raise ValueError("No scoring_results found in context. Run ScoreModelsStep first.")

        if entity_cols:
            scored_df = data[entity_cols].copy()
        else:
            # Fall back to an empty DF keyed on the current index if no entities configured.
            scored_df = pd.DataFrame(index=data.index)

        for model_key, scores in scoring_results.items():
            if not isinstance(scores, pd.Series):
                raise TypeError(f"Expected pd.Series for model_key={model_key!r}, got {type(scores)}")
            if not scores.index.equals(scored_df.index):
                raise ValueError(
                    f"Scores index for model_key={model_key!r} does not match base index. "
                    "Ensure scoring uses the same X_shared index for all models in the model group."
                )
            scored_df[model_key] = scores.astype("float64")

        context["scored_df"] = scored_df
        context["status"] = "combined"
        return context


class PersistScoringResultsStep:
    """
    Persist the combined scoring results DataFrame to the results table in the sql database.
    """
    def __call__(self, context: ModelContext) -> ModelContext:
        logger.info("Persisting scoring results")

        model_specs: dict = context["model_specs"]
        sql_client: SqlClient = context["sql_client"]
        system_event_id = context["system_event_id"]
        model_group_key = context["model_group_key"]

        scored_df = context.get("scored_df")
        if scored_df is None or not isinstance(scored_df, pd.DataFrame):
            raise ValueError("PersistScoringResultsStep expected context['scored_df'] as a pandas DataFrame")
        
        results_table_name = model_specs.get("results_table_name")
        if not results_table_name:
            raise ValueError("Model spec missing 'results_table_name' for persisting scoring results")

        logger.info("Persisting results for model_group_key=%s", model_group_key)

        # add metadata columns to the scored dataframe
        scored_df["SystemEventId"] = system_event_id
        scored_df["ScoredAtUtc"] = datetime.now(timezone.utc)
        scored_df["ModelGroupKey"] = model_group_key
    
        # schema check: ensure the scored_df columns match the expected schema in the results table
        target_schema = sql_client.get_schema(
        table_name=results_table_name
        )

        # convert the schema to a richer dictionary keyed by column name (accepted by validate_transformed_data )
        schema_dict = {
            "columns": [col["column_name"] for col in target_schema],
            "data_types": {col["column_name"]: col["data_type"] for col in target_schema},
            "required": {col["column_name"]: bool(col["is_required"]) for col in target_schema},
        }

        validate_transformed_data(transformed_data=scored_df, target_schema=schema_dict)

        try:
            sql_client.write_dataframe_to_table(
                df=scored_df,
                table_name=results_table_name,
                if_exists="append",
            )

        except Exception as e:
            logger.error("Error persisting scores for model_group_key=%s: %s", model_group_key, e)
            raise

        context["status"] = "persisted"
        return context

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
        PrepareTrainingContextStep(),
        BuildTrainingPayloadsStep(),
        TrainModelsStep(),
        EvaluateModelsStep(),
        PersistModelArtifactsStep(),
    )
    return pipeline_steps

### factory for the default model scoring pipeline ###
def default_model_scoring_pipeline_factory() -> tuple[ModelStep, ...]:
    """
    Default end‑to‑end orchestration for model scoring:
        1. Start the scoring run
        2. Load the model specification
        3. Load the scoring data
        4. Prepare the scoring context
        5. Load the model artifacts
        6. Score the model
        7. Persist the scoring results
        """

    pipeline_steps = (
        StartScoringRunStep(),
        LoadModelSpecStep(),
        LoadDataStep(),
        PrepareScoringContextStep(),
        LoadModelArtifactsStep(),
        ScoreModelsStep(),
        CombineScoringResultsStep(),
        PersistScoringResultsStep(),
    )
    return pipeline_steps


# Register the default orchestration pipeline at import time.
#NOTE: Register the pipelines here if you want to use them in the orchestrator
register_model_pipeline("default_model_training", default_model_training_pipeline_factory)
register_model_pipeline("default_model_scoring", default_model_scoring_pipeline_factory)    