"""HTTP request and response models for the inference API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BatchPredictionRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        min_length=1,
        description="Raw feature records. Every record must follow the artifact feature contract.",
    )


class PredictionItem(BaseModel):
    prediction: Any = Field(description="Predicted class or regression value.")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Class probabilities when the fitted estimator supports predict_proba.",
    )


class PredictionResponse(PredictionItem):
    run_id: str = Field(description="Immutable training run that produced the artifact.")
    executed_at: datetime = Field(description="UTC timestamp when inference was executed.")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionItem] = Field(description="Predictions in the same order as the input instances.")
    run_id: str = Field(description="Immutable training run that produced the artifact.")
    executed_at: datetime = Field(description="UTC timestamp when inference was executed.")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
