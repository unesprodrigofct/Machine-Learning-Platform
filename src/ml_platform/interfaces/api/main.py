"""FastAPI adapter for model inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ml_platform.adapters.artifact_stores.artifact_loader import ArtifactLoader
from ml_platform.application.inference import InferenceService
from ml_platform.domain.errors import (
    ArtifactError,
    DataValidationError,
    PlatformError,
    TrainingError,
)
from ml_platform.interfaces.api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionResponse,
)


def create_app(artifact_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Machine Learning Platform Inference API",
        version="0.1.0",
        description="Serve predictions from a versioned fitted inference pipeline.",
    )
    configured_path = artifact_path or os.getenv("MODEL_ARTIFACT_PATH")
    app.state.inference_service = None
    app.state.artifact_error = None
    if configured_path:
        try:
            app.state.inference_service = InferenceService(ArtifactLoader(configured_path).load())
        except ArtifactError as error:
            app.state.artifact_error = error

    @app.exception_handler(PlatformError)
    async def platform_error_handler(_: Request, error: PlatformError) -> JSONResponse:
        if isinstance(error, ArtifactError):
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
            code = "model_not_ready"
        elif isinstance(error, DataValidationError):
            http_status = status.HTTP_400_BAD_REQUEST
            code = "invalid_request"
        elif isinstance(error, TrainingError):
            http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
            code = "prediction_error"
        else:
            http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
            code = "internal_error"
        return JSONResponse(
            status_code=http_status,
            content={"error": {"code": code, "message": error.message, "details": error.details}},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        service = _service(app)
        metadata = service.context.metadata
        return {
            "status": "ready",
            "run_id": service.context.run_id,
            "artifact_version": metadata.get("artifact_version", service.context.run_id),
            "model_type": metadata.get("model_type"),
            "task": metadata.get("task"),
            "algorithm": metadata.get("algorithm"),
        }

    @app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    async def predict(features: dict[str, Any]) -> dict[str, Any]:
        service = _service(app)
        result = service.predict_one(features)
        return {**result, "run_id": service.context.run_id}

    @app.post("/predict/batch", response_model=BatchPredictionResponse, responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    async def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
        service = _service(app)
        return {"predictions": service.predict_batch(request.instances), "run_id": service.context.run_id}

    return app


def _service(app: FastAPI) -> InferenceService:
    if app.state.inference_service is None:
        error = app.state.artifact_error or ArtifactError("MODEL_ARTIFACT_PATH is not configured.")
        raise error
    return app.state.inference_service


app = create_app()
