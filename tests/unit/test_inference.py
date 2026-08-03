from pathlib import Path

import pandas as pd
import pytest

from ml_platform.adapters.artifact_stores.artifact_loader import ArtifactContext
from ml_platform.application.inference import InferenceEngine, InferenceService
from ml_platform.domain.errors import DataValidationError, TrainingError


class PredictOnlyPipeline:
    def predict(self, features: pd.DataFrame) -> list[int]:
        return [1] * len(features)


class FailingPipeline:
    def predict(self, _: pd.DataFrame) -> list[int]:
        raise RuntimeError("prediction failed")


def _context(pipeline: object, schema: dict[str, str] | None = None) -> ArtifactContext:
    return ArtifactContext(
        pipeline=pipeline,
        schema=schema or {"feature": "float64"},
        metadata={"run_id": "run-123"},
        artifact_path=Path("artifacts/run-123"),
    )


def test_inference_engine_supports_estimators_without_probabilities() -> None:
    engine = InferenceEngine()

    result = engine.predict(_context(PredictOnlyPipeline()), pd.DataFrame({"feature": [1.0]}))

    assert result == [{"prediction": 1}]


def test_inference_engine_wraps_pipeline_failures() -> None:
    with pytest.raises(TrainingError, match="could not be completed"):
        InferenceEngine().predict(_context(FailingPipeline()), pd.DataFrame({"feature": [1.0]}))


def test_inference_service_rejects_empty_batch() -> None:
    with pytest.raises(DataValidationError, match="Batch must contain"):
        InferenceService(_context(PredictOnlyPipeline())).predict_batch([])


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ({}, "Required feature"),
        ({"feature": 1, "extra": 2}, "Unknown feature"),
        ({"feature": "not-a-number"}, "invalid type"),
    ],
)
def test_inference_service_rejects_invalid_feature_contract(features: dict[str, object], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        InferenceService(_context(PredictOnlyPipeline())).predict_one(features)
