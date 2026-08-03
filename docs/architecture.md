# Architecture Overview

The platform is a modular monolith built around adapters and plugins, deployed
locally with Docker Compose and proposed for AWS deployment in
[aws-architecture.drawio](aws-architecture.drawio).

```text
CLI / REST API adapters
          |
          v
Application services (training and inference use cases)
          |
          +--> Ports --> Data source and artifact-store adapters
          |
          +--> Plugin registry --> Estimators, transformers, evaluators, validators
```

## AWS deployment proposal

The local Docker Compose workflow maps to the following AWS components:

- Amazon S3 stores versioned datasets and immutable inference artifacts.
- SageMaker Training Jobs or ECS/Fargate tasks execute configuration-driven training.
- SageMaker Model Registry tracks approved model versions.
- API Gateway exposes the REST/OpenAPI contract.
- AWS Lambda serves the small tabular model used in this case.
- ECS/Fargate or a SageMaker Endpoint is the alternative for larger models, heavier dependencies, or stricter latency requirements.
- IAM provides least-privilege access to data and artifacts.
- Secrets Manager stores database credentials.
- CloudWatch collects logs, metrics, and alarms.
- ECR stores versioned container images.

The deployed inference service always loads one explicit artifact version. It
does not train at request time or select a mutable latest model implicitly.

## Deployment flow

```text
Code change
    |
    v
CI/CD tests and builds image
    |
    v
ECR + training job
    |
    v
Versioned S3 artifact + model registry approval
    |
    v
API Gateway -> Lambda / ECS / SageMaker Endpoint
```

Lambda is appropriate for the small scikit-learn tabular model used in this
case. For production models with large artifacts or strict latency targets,
the same inference contract can be deployed to ECS/Fargate or a SageMaker
Endpoint without changing the REST API contract.
