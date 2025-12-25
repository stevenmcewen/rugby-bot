from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set

from xgboost import XGBClassifier, XGBRegressor


ModelBuilder = Callable[[dict], Any]

# Business model keys -> allowed prediction types
MODEL_COMPATIBILITY: dict[str, Set[str]] = {
    "international_rugby_homewin_v1": {"classification"},
    "international_rugby_pointdiff_v1": {"regression"},
}


MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(model_key: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if model_key in MODEL_REGISTRY:
            raise KeyError(f"Duplicate model_key registered: {model_key}")
        MODEL_REGISTRY[model_key] = builder
        return builder
    return decorator


@register_model("international_rugby_homewin_v1")
def build_xgb_classifier(params: dict) -> Any:
    return XGBClassifier(**params)


@register_model("international_rugby_pointdiff_v1")
def build_xgb_regressor(params: dict) -> Any:
    return XGBRegressor(**params)


@dataclass(frozen=True)
class ModelBuildContext:
    model_key: str
    prediction_type: str 
    model_params: Optional[dict] = None


class ModelFactory:
    @staticmethod
    def create(ctx: ModelBuildContext) -> Any:
        # 1) Validate model key exists in registry
        if ctx.model_key not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY.keys()))
            raise KeyError(f"Unknown model_key='{ctx.model_key}'. Available: {available}")

        # 2) Validate prediction type compatibility
        normalized_pt = normalize_prediction_type(ctx.prediction_type)

        allowed = MODEL_COMPATIBILITY.get(ctx.model_key)
        if allowed is not None and normalized_pt not in allowed:
            raise ValueError(
                f"Model '{ctx.model_key}' not compatible with prediction_type='{ctx.prediction_type}'. "
                f"Allowed: {sorted(allowed)}"
            )

        params: dict = {}
        if ctx.model_params:
            # check that params are in the correct format
            if not isinstance(ctx.model_params, dict):
                raise ValueError("model_params must be a dictionary")
            params.update(ctx.model_params)

        # Build model
        builder = MODEL_REGISTRY[ctx.model_key]
        model = builder(params)
        return model


# helpers
def normalize_prediction_type(prediction_type: str) -> str:
    pt = (prediction_type or "").strip().lower()
    if pt in {"binary", "classification", "classify"}:
        return "classification"
    if pt in {"regression", "regress"}:
        return "regression"
    raise ValueError(f"Unknown prediction_type='{prediction_type}'")