# Architecture Overview

The platform is a modular monolith built around adapters and plugins.

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

The training capability produces a versioned artifact containing the fitted preprocessing pipeline, estimator, schema, configuration, metrics, and run metadata. The future inference API consumes this artifact without duplicating feature logic.
