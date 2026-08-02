from pathlib import Path

import pandas as pd

from ml_platform.application.train_model import train
from ml_platform.domain.config import TrainingConfig
import yaml


def test_training_persists_a_portable_artifact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "training.csv"
    dataframe = pd.DataFrame({
        "distance_km": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "weather": ["clear", "rain"] * 6,
        "late": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    })
    dataframe.to_csv(dataset_path, index=False)
    config = TrainingConfig.model_validate({
        "name": "integration-test",
        "problem_type": "supervised",
        "task": "classification",
        "data_source": {"type": "csv", "path": str(dataset_path)},
        "target_column": "late",
        "validation": {"strategy": "out_of_sample", "test_size": 0.25},
        "model": {"algorithm": "logistic_regression"},
        "artifacts": {"root_path": str(tmp_path / "artifacts")},
    })

    result = train(config)

    artifact_path = Path(result.artifact_path)
    assert (artifact_path / "inference_pipeline.joblib").exists()
    assert (artifact_path / "metrics.json").exists()
    assert (artifact_path / "schema.json").exists()
    assert set(result.metrics) == {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
        "ks",
    }
    persisted_config = yaml.safe_load((artifact_path / "config.yaml").read_text())
    assert persisted_config["evaluation"]["metrics"] == list(result.metrics)
    metadata = yaml.safe_load((artifact_path / "metadata.json").read_text())
    assert metadata["artifact_version"] == result.run_id
    assert metadata["model_type"] == "sklearn_tabular"
    assert "training_code_version" in metadata
