# Platform Architecture

The Machine Learning Platform is organized around two independent capabilities:

```text
Training Capability  ->  Versioned Artifact  ->  Inference Capability
```

The architectural principle is:

> The inference capability must consume the fitted pipeline produced by the
> training capability. It must never rebuild or reimplement preprocessing.

This preserves one training-serving contract for feature validation,
preprocessing, model inference, and model version selection.

## Capability boundaries

### Training capability

The training capability is configuration-driven. It reads data and training
configuration, resolves plugins, fits preprocessing and the estimator, evaluates
the result, and writes an immutable artifact.

```text
Training configuration
        |
        v
Data source adapter -> Validation -> Fitted preprocessing + estimator
                                               |
                                               v
                              Versioned inference artifact
```

### Artifact

The artifact is the boundary between training and inference:

```text
artifacts/<run_id>/
├── inference_pipeline.joblib  # fitted preprocessing and estimator
├── schema.json                # feature contract
├── metadata.json              # model and lineage metadata
├── config.yaml                # effective training configuration
└── metrics.json               # validation report
```

`inference_pipeline.joblib` contains the learned imputer, encoder, scaler, and
estimator when those components are configured. The inference service receives
raw feature values and delegates transformations to this fitted pipeline.

### Inference capability

The inference capability loads one explicit artifact and exposes a REST
interface:

```text
JSON request
    |
    v
HTTP validation
    |
    v
Inference Service
    |
    +--> Artifact Loader
    |       ├── schema.json
    |       ├── metadata.json
    |       └── inference_pipeline.joblib
    |
    v
Inference validation against schema
    |
    v
Fitted pipeline.predict()
    |
    v
Prediction + run_id + executed_at
```

HTTP validation and inference validation have separate responsibilities:

- HTTP validation checks JSON shape, batch structure, and request serialization.
- Inference validation checks feature names and values against `schema.json`.
- Preprocessing semantics belong exclusively to the fitted pipeline.

## Local deployment with Docker Compose

The local environment contains two services:

```text
                       shared repository volumes
                 ┌────────────────────────────────┐
                 │ configs / data / artifacts      │
                 └───────────────┬────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
       ┌──────▼──────┐                       ┌──────▼──────┐
       │   training  │                       │  inference  │
       │  one-shot   │                       │ long-lived  │
       └──────┬──────┘                       └──────┬──────┘
              │ writes artifact                     │ reads selected artifact
              └──────────────► artifacts/<run_id> ◄──┘
                                                       │
                                                       ▼
                                                REST :8000
```

Training and inference are deliberately separate lifecycles. Starting the
inference service does not trigger training. The artifact is selected through
`MODEL_ARTIFACT_PATH`, which makes the deployed model version explicit.

## End-to-end operating flow

```text
1. Generate or ingest data
2. Run the training capability
3. Persist a versioned artifact
4. Select the artifact run_id
5. Start the inference capability
6. Validate readiness
7. Send raw features through REST
8. Return prediction, run_id, and executed_at
```

The complete local walkthrough is maintained in private developer notes and is
intentionally excluded from the repository.

## API documentation

When the inference service is running, the interactive Swagger UI is available
at:

```text
http://localhost:8000/docs
```

The machine-readable OpenAPI contract is available at:

```text
http://localhost:8000/openapi.json
```

The API documents successful responses and the structured error contract for
invalid requests, prediction failures, and unavailable artifacts.

## AWS deployment proposal

The public deployment diagram is available in
[aws-architecture.drawio](aws-architecture.drawio). It intentionally uses
generic service names so AWS icons can be added during presentation.

```text
Code repository
      |
      v
CI/CD -> container registry
      |
      +--> Training job (SageMaker or ECS/Fargate)
      |          |
      |          v
      |     Versioned artifact in S3
      |          |
      |          v
      |     Model Registry approval
      |
      +--> Inference deployment
                 |
                 +--> Lambda for this small tabular model
                 +--> ECS/Fargate for a long-running API
                 +--> SageMaker Endpoint for managed model serving
                             |
                             v
                  API Gateway / load balancer
```

Proposed AWS responsibilities:

- S3: versioned datasets and immutable artifacts.
- SageMaker Training or ECS/Fargate: training execution.
- SageMaker Model Registry: approval and promotion of model versions.
- Lambda, ECS/Fargate, or SageMaker Endpoint: inference runtime.
- API Gateway or Application Load Balancer: public HTTP entry point.
- ECR: versioned container images.
- IAM: least-privilege access to data and artifacts.
- Secrets Manager: database credentials and external secrets.
- CloudWatch: logs, metrics, health, and alarms.

Lambda is suitable for the small scikit-learn tabular model in this case. The
same REST and artifact contracts can be deployed to ECS/Fargate or SageMaker
when model size, startup time, throughput, or latency requirements change.
