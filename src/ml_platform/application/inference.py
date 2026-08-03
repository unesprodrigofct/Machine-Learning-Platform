"""Inference use case independent from HTTP and filesystem concerns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_platform.adapters.artifact_stores.artifact_loader import ArtifactContext
from ml_platform.domain.errors import DataValidationError, TrainingError


class InferenceEngine:
    """Execute predictions against a fitted inference pipeline."""

    def predict(self, context: ArtifactContext, features: pd.DataFrame) -> list[dict[str, Any]]:
        try:
            predictions = context.pipeline.predict(features)
            probabilities = self._probabilities(context.pipeline, features)
        except Exception as error:
            raise TrainingError("Prediction could not be completed.") from error

        results: list[dict[str, Any]] = []
        for index, prediction in enumerate(predictions):
            result: dict[str, Any] = {"prediction": _json_value(prediction)}
            if probabilities is not None:
                result["probabilities"] = {
                    str(_json_value(label)): _json_value(probabilities[index][class_index])
                    for class_index, label in enumerate(self._classes(context.pipeline, probabilities.shape[1]))
                }
            results.append(result)
        return results

    @staticmethod
    def _probabilities(pipeline: Any, features: pd.DataFrame) -> np.ndarray | None:
        if not hasattr(pipeline, "predict_proba"):
            return None
        probabilities = np.asarray(pipeline.predict_proba(features))
        if probabilities.ndim != 2:
            return None
        return probabilities

    @staticmethod
    def _classes(pipeline: Any, count: int) -> list[Any]:
        classes = getattr(pipeline, "classes_", None)
        if classes is None and hasattr(pipeline, "named_steps"):
            estimator = pipeline.named_steps.get("model")
            classes = getattr(estimator, "classes_", None)
        if classes is None:
            return list(range(count))
        return list(classes)


class InferenceService:
    """Coordinate feature-contract validation and prediction."""

    def __init__(self, context: ArtifactContext, engine: InferenceEngine | None = None) -> None:
        self._context = context
        self._engine = engine or InferenceEngine()

    @property
    def context(self) -> ArtifactContext:
        return self._context

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        results = self.predict_batch([features])
        return results[0]

    def predict_batch(self, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not instances:
            raise DataValidationError("Batch must contain at least one instance.")
        dataframe = self._validate_features(instances)
        return self._engine.predict(self._context, dataframe)

    def _validate_features(self, instances: list[dict[str, Any]]) -> pd.DataFrame:
        expected = list(self._context.schema)
        expected_set = set(expected)
        for index, instance in enumerate(instances):
            received = set(instance)
            missing = [name for name in expected if name not in received]
            extra = sorted(received - expected_set)
            if missing:
                raise DataValidationError(
                    "Required feature is missing.", details={"index": index, "fields": missing}
                )
            if extra:
                raise DataValidationError(
                    "Unknown feature was provided.", details={"index": index, "fields": extra}
                )
        dataframe = pd.DataFrame(instances, columns=expected)
        for index, instance in enumerate(instances):
            for column, dtype in self._context.schema.items():
                if not _values_match_dtype(pd.Series([instance[column]]), dtype):
                    raise DataValidationError(
                        f"Feature {column!r} has an invalid type.",
                        details={"index": index, "field": column, "expected_type": dtype},
                    )
        return dataframe


def _values_match_dtype(values: pd.Series, dtype: str) -> bool:
    non_null = values.dropna().tolist()
    if not non_null:
        return True
    if "int" in dtype or "float" in dtype:
        return all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null)
    if "bool" in dtype:
        return all(isinstance(value, bool) for value in non_null)
    return all(isinstance(value, str) for value in non_null)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
