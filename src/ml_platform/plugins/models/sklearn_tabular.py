"""Reference plugin declaring the tabular scikit-learn capability."""

from ml_platform.domain.config import TrainingConfig
from ml_platform.plugins.estimators.registry import create_estimator


class SklearnTabularModelPlugin:
    modality = "tabular"
    model_type = "sklearn_tabular"

    def supports(self, task: str, algorithm: str) -> bool:
        try:
            create_estimator(algorithm, task, {}, 42)
        except Exception:
            return False
        return True

    def capabilities(self) -> dict[str, object]:
        return {"model_type": self.model_type, "modality": self.modality, "framework": "scikit-learn", "tasks": ["classification", "regression", "clustering", "anomaly_detection"]}

    def validate(self, config: TrainingConfig) -> None:
        if not self.supports(config.task, config.model.algorithm):
            raise ValueError(f"Unsupported scikit-learn algorithm/task combination: {config.model.algorithm}/{config.task}")
