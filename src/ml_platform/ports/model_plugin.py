"""Contract implemented by framework- and modality-specific model plugins."""

from typing import Any, Protocol

from ml_platform.domain.config import TrainingConfig
from ml_platform.domain.models import DatasetSplit


class ModelPlugin(Protocol):
    modality: str
    model_type: str

    def supports(self, task: str, algorithm: str) -> bool: ...
    def train(self, split: DatasetSplit, config: TrainingConfig) -> Any: ...
    def evaluate(self, trained_model: Any, validation_data: Any, config: TrainingConfig) -> dict[str, float]: ...
    def inference_schema(self, training_data: Any, config: TrainingConfig) -> dict[str, str]: ...
