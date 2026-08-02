"""Built-in estimator plugin registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml_platform.domain.errors import PluginResolutionError

EstimatorFactory = Callable[[dict[str, Any]], Any]


def _logistic_regression(parameters: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, **parameters)


def _random_forest_classifier(parameters: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(**parameters)


def _linear_regression(parameters: dict[str, Any]) -> LinearRegression:
    return LinearRegression(**parameters)


def _random_forest_regressor(parameters: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(**parameters)


def _kmeans(parameters: dict[str, Any]) -> KMeans:
    return KMeans(**parameters)


def _isolation_forest(parameters: dict[str, Any]) -> IsolationForest:
    return IsolationForest(**parameters)


def _xgb_classifier(parameters: dict[str, Any]) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise PluginResolutionError("XGBoost support requires `pip install .[xgboost]`.") from error
    return XGBClassifier(**parameters)


def _xgb_regressor(parameters: dict[str, Any]) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise PluginResolutionError("XGBoost support requires `pip install .[xgboost]`.") from error
    return XGBRegressor(**parameters)


_REGISTRY: dict[str, tuple[set[str], EstimatorFactory]] = {
    "logistic_regression": ({"classification"}, _logistic_regression),
    "random_forest_classifier": ({"classification"}, _random_forest_classifier),
    "linear_regression": ({"regression"}, _linear_regression),
    "random_forest_regressor": ({"regression"}, _random_forest_regressor),
    "kmeans": ({"clustering"}, _kmeans),
    "isolation_forest": ({"anomaly_detection"}, _isolation_forest),
    "xgboost_classifier": ({"classification"}, _xgb_classifier),
    "xgboost_regressor": ({"regression"}, _xgb_regressor),
}


def create_estimator(algorithm: str, task: str, parameters: dict[str, Any], random_seed: int) -> Any:
    try:
        supported_tasks, factory = _REGISTRY[algorithm]
    except KeyError as error:
        supported = ", ".join(sorted(_REGISTRY))
        raise PluginResolutionError(f"Unknown estimator {algorithm!r}. Supported estimators: {supported}.") from error
    if task not in supported_tasks:
        raise PluginResolutionError(f"Estimator {algorithm!r} is not compatible with task {task!r}.")
    resolved_parameters = dict(parameters)
    if algorithm in {"random_forest_classifier", "random_forest_regressor", "kmeans", "isolation_forest"}:
        resolved_parameters.setdefault("random_state", random_seed)
    return factory(resolved_parameters)
