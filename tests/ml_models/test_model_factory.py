import pytest

from functions.ml_models.helpers.model_factory import ModelFactory, ModelBuildContext, normalize_prediction_type

def test_normalize_prediction_type_maps_common_aliases():
    assert normalize_prediction_type("binary") == "classification"
    assert normalize_prediction_type("classification") == "classification"
    assert normalize_prediction_type("classify") == "classification"
    assert normalize_prediction_type("regression") == "regression"
    assert normalize_prediction_type("regress") == "regression"


def test_normalize_prediction_type_raises_on_unknown():
    with pytest.raises(ValueError):
        normalize_prediction_type("weird")


def test_model_factory_raises_on_unknown_model_key():
    with pytest.raises(KeyError) as exc:
        ModelFactory.create(ModelBuildContext(model_key="unknown", prediction_type="classification"))
    assert "Unknown model_key" in str(exc.value)


def test_model_factory_validates_prediction_type_compatibility():
    # international_rugby_homewin_v1 only allows classification
    with pytest.raises(ValueError) as exc:
        ModelFactory.create(ModelBuildContext(model_key="international_rugby_homewin_v1", prediction_type="regression"))
    assert "not compatible" in str(exc.value).lower()


def test_model_factory_rejects_non_dict_params():
    with pytest.raises(ValueError):
        ModelFactory.create(  # type: ignore[arg-type]
            ModelBuildContext(model_key="international_rugby_homewin_v1", prediction_type="classification", model_params="x")
        )


