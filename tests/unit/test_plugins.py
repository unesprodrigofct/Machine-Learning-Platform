import pytest

from ml_platform.domain.errors import PluginResolutionError
from ml_platform.plugins.estimators.registry import create_estimator


def test_estimator_registry_creates_compatible_model() -> None:
    model = create_estimator("random_forest_classifier", "classification", {"n_estimators": 5}, 42)

    assert model.random_state == 42
    assert model.n_estimators == 5


def test_estimator_registry_rejects_incompatible_task() -> None:
    with pytest.raises(PluginResolutionError, match="not compatible"):
        create_estimator("kmeans", "classification", {}, 42)


def test_estimator_registry_rejects_unknown_algorithm() -> None:
    with pytest.raises(PluginResolutionError, match="Unknown estimator"):
        create_estimator("unknown", "classification", {}, 42)
