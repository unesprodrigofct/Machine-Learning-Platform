"""HTTP request and response models for the inference API."""

from typing import Any

from pydantic import BaseModel, Field


class BatchPredictionRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(min_length=1)


class PredictionItem(BaseModel):
    prediction: Any
    probabilities: dict[str, float] | None = None


class PredictionResponse(PredictionItem):
    run_id: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionItem]
    run_id: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
