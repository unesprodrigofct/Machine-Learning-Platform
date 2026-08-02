# Machine Learning Platform

A configurable and reproducible platform for training, evaluating, versioning, and serving supervised and unsupervised machine learning models.

## Status

The repository scaffold and the training capability specification are in place. The first implementation milestone is the configuration-driven training workflow.

## Architecture principles

- **Configuration first:** Every run is defined by a validated YAML configuration.
- **Reproducible by default:** Seeds, effective configuration, data fingerprints, metrics, and environment metadata are persisted with every model artifact.
- **Adapters at the boundary:** Data sources and artifact stores can change without affecting training orchestration.
- **Plugins for ML variability:** Algorithms, feature transformers, evaluators, and validation strategies are registered behind explicit contracts.
- **Training-serving consistency:** A fitted preprocessing pipeline and estimator are serialized together.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/ml_platform/` | Platform source code |
| `configs/training/` | Versioned training configurations |
| `sql/` | Read-only source extraction and enrichment queries |
| `data/sample/` | Small non-sensitive example data |
| `artifacts/` | Local generated model artifacts (not versioned) |
| `tests/` | Unit and integration tests |
| `docs/` | Architecture and operational documentation |

## Documentation

- [Training capability specification](.vscode/training-platform-spec.md)
- [Inference REST API specification](.vscode/inference-api-spec.md)
- [Inference and infrastructure code walkthrough](.vscode/inference-infrastructure-code-walkthrough.md)
- [Local infrastructure specification](.vscode/infrastructure-spec.md)
- [Architecture overview](docs/architecture.md)

## Docker Compose execution

Docker Compose is the primary execution path for the case. The same image
supports both the one-shot training job and the long-running inference API.

Build the image:

```bash
docker compose -f docker/docker-compose.yml build
```

Generate the deterministic example dataset on the host, then train through
Compose:

```bash
python3 scripts/generate_sample_data.py
docker compose -f docker/docker-compose.yml run --rm training \
  --config /app/configs/training/classification.yaml
```

The generated artifact remains in the host `artifacts/` directory. Set the
artifact version explicitly before starting the API:

```bash
export MODEL_ARTIFACT_PATH=/app/artifacts/<run_id>
docker compose -f docker/docker-compose.yml up inference
```

The API is available at `http://localhost:8000/docs`.

For local development without containers, the following workflow is also
supported:

## Local Python execution

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[api,dev]"
```

Generate the deterministic example dataset and train the example model:

```bash
python3 scripts/generate_sample_data.py
.venv/bin/ml-platform --config configs/training/classification.yaml
```

Each run writes a self-contained artifact directory under `artifacts/`.

Evaluation metrics are selected from `evaluation.metrics` in the training
configuration. When that field is omitted, the task-specific defaults from
`configs/metrics/defaults.yaml` are used. For example, the included
classification configuration uses all defaults defined for classification.

Run the automated tests with:

```bash
.venv/bin/pytest
```

## Serving the trained model

The API is an optional capability. Point it to one immutable training artifact
version and start Uvicorn:

```bash
MODEL_ARTIFACT_PATH=artifacts/<run_id> \
.venv/bin/uvicorn ml_platform.interfaces.api.main:app --host 0.0.0.0 --port 8000
```

The artifact contains the fitted inference pipeline, feature contract, and
metadata. The API does not retrain the model or duplicate preprocessing.

Open Swagger at `http://localhost:8000/docs`.

Check readiness:

```bash
curl http://localhost:8000/ready
```

Send one prediction using the raw feature values expected by the training
schema:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km": 8.5, "weather": "rain"}'
```

The API returns the prediction and the immutable `run_id` of the loaded
artifact.
