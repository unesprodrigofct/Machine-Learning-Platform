"""Task-aware metric registry and evaluation functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
    roc_auc_score,
)

from ml_platform.domain.errors import ConfigurationError


def _ks_score(target: pd.Series, scores: np.ndarray) -> float:
    positive = np.sort(scores[np.asarray(target) == 1])
    negative = np.sort(scores[np.asarray(target) == 0])
    if not len(positive) or not len(negative):
        return float("nan")
    thresholds = np.sort(np.unique(np.concatenate([positive, negative])))
    positive_cdf = np.searchsorted(positive, thresholds, side="right") / len(positive)
    negative_cdf = np.searchsorted(negative, thresholds, side="right") / len(negative)
    return float(np.max(np.abs(positive_cdf - negative_cdf)))


def _classification_scores(pipeline: Any, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        return probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(features))
    return None


def evaluate(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series | None,
    task: str,
    requested_metrics: list[str],
) -> dict[str, float]:
    predictions = pipeline.predict(features)
    metric_names = requested_metrics
    if task == "classification":
        assert target is not None
        scores = _classification_scores(pipeline, features)
        values: dict[str, float] = {}
        for metric in metric_names:
            if metric == "accuracy":
                values[metric] = float(accuracy_score(target, predictions))
            elif metric == "precision_weighted":
                values[metric] = float(precision_score(target, predictions, average="weighted", zero_division=0))
            elif metric == "recall_weighted":
                values[metric] = float(recall_score(target, predictions, average="weighted", zero_division=0))
            elif metric == "f1_weighted":
                values[metric] = float(f1_score(target, predictions, average="weighted", zero_division=0))
            elif metric == "matthews_corrcoef":
                values[metric] = float(matthews_corrcoef(target, predictions))
            elif metric in {"roc_auc", "ks"}:
                if scores is None or scores.ndim != 1:
                    values[metric] = float("nan")
                elif metric == "roc_auc":
                    values[metric] = float(roc_auc_score(target, scores))
                else:
                    values[metric] = _ks_score(target, scores)
            else:
                raise ConfigurationError(f"Unsupported classification metric: {metric}")
        return values
    if task == "regression":
        assert target is not None
        values = {}
        for metric in metric_names:
            if metric == "mae":
                values[metric] = float(mean_absolute_error(target, predictions))
            elif metric == "rmse":
                values[metric] = float(mean_squared_error(target, predictions) ** 0.5)
            elif metric == "r2":
                values[metric] = float(r2_score(target, predictions))
            elif metric == "mape":
                values[metric] = float(mean_absolute_percentage_error(target, predictions))
            else:
                raise ConfigurationError(f"Unsupported regression metric: {metric}")
        return values
    if task == "clustering":
        transformed = pipeline.named_steps["preprocessor"].transform(features)
        values = {}
        for metric in metric_names:
            if metric == "silhouette_score":
                values[metric] = float(silhouette_score(transformed, predictions))
            elif metric == "calinski_harabasz_score":
                values[metric] = float(calinski_harabasz_score(transformed, predictions))
            elif metric == "davies_bouldin_score":
                values[metric] = float(davies_bouldin_score(transformed, predictions))
            else:
                raise ConfigurationError(f"Unsupported clustering metric: {metric}")
        return values
    if task == "anomaly_detection":
        values = {}
        for metric in metric_names:
            if metric == "anomaly_rate":
                values[metric] = float(np.mean(predictions == -1))
            else:
                raise ConfigurationError(f"Unsupported anomaly-detection metric: {metric}")
        return values
    raise ConfigurationError(f"Unsupported evaluation task: {task}")
