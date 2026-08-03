import asyncio
from pathlib import Path

import httpx2
import pandas as pd

from ml_platform.application.train_model import train
from ml_platform.domain.config import TrainingConfig
from ml_platform.interfaces.api.main import create_app


def _trained_artifact(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "training.csv"
    pd.DataFrame({
        "distance_km": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "weather": ["clear", "rain"] * 6,
        "late": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    }).to_csv(dataset_path, index=False)
    config = TrainingConfig.model_validate({
        "name": "api-integration-test",
        "problem_type": "supervised",
        "task": "classification",
        "data_source": {"type": "csv", "path": str(dataset_path)},
        "target_column": "late",
        "validation": {"strategy": "out_of_sample", "test_size": 0.25},
        "model": {"algorithm": "logistic_regression"},
        "artifacts": {"root_path": str(tmp_path / "artifacts")},
    })
    return Path(train(config).artifact_path)


def test_api_serves_health_readiness_prediction_and_openapi(tmp_path: Path) -> None:
    artifact_path = _trained_artifact(tmp_path)
    asyncio.run(_exercise_successful_api(create_app(artifact_path)))


async def _exercise_successful_api(app) -> None:
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        cors = await client.get("/health", headers={"Origin": "http://localhost:8000"})
        assert cors.headers["access-control-allow-origin"] == "http://localhost:8000"
        readiness = await client.get("/ready")
        assert readiness.status_code == 200
        assert readiness.json()["model_type"] == "sklearn_tabular"

        prediction = await client.post("/predict", json={"distance_km": 8, "weather": "rain"})
        assert prediction.status_code == 200
        assert prediction.json()["run_id"] == readiness.json()["run_id"]
        assert "prediction" in prediction.json()
        assert "probabilities" in prediction.json()
        assert "executed_at" in prediction.json()

        batch = await client.post(
            "/predict/batch",
            json={"instances": [
                {"distance_km": 8, "weather": "rain"},
                {"distance_km": 2, "weather": "clear"},
            ]},
        )
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 2
        assert "executed_at" in batch.json()
        openapi = await client.get("/openapi.json")
        assert openapi.status_code == 200
        docs = await client.get("/docs")
        assert docs.status_code == 200
        assert "/swagger.css" in docs.text
        css = await client.get("/swagger.css")
        assert css.status_code == 200
        assert "API RESPONSE" in css.text
        assert openapi.json()["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"]["example"] == {"status": "ok"}
        assert openapi.json()["paths"]["/ready"]["get"]["responses"]["200"]["content"]["application/json"]["example"]["status"] == "ready"
        assert openapi.json()["paths"]["/predict"]["post"]["responses"]["200"]["content"]["application/json"]["example"]["run_id"]
        assert openapi.json()["paths"]["/predict"]["post"]["responses"]["400"]["content"]["application/json"]["example"]["error"]["code"] == "invalid_request"


def test_api_rejects_invalid_feature_contract(tmp_path: Path) -> None:
    asyncio.run(_exercise_invalid_request(create_app(_trained_artifact(tmp_path))))


async def _exercise_invalid_request(app) -> None:
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"distance_km": 8})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"
        assert response.json()["error"]["details"]["fields"] == ["weather"]


def test_api_reports_not_ready_when_artifact_is_missing(tmp_path: Path) -> None:
    asyncio.run(_exercise_not_ready(create_app(tmp_path / "missing-artifact")))


async def _exercise_not_ready(app) -> None:
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_ready"
