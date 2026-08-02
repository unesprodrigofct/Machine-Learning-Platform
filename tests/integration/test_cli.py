from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from ml_platform.interfaces.cli.main import app


def test_cli_runs_a_training_job(tmp_path: Path) -> None:
    dataset_path = tmp_path / "training.csv"
    pd.DataFrame({
        "feature": list(range(12)),
        "category": ["a", "b"] * 6,
        "target": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    }).to_csv(dataset_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "name": "cli-test",
        "problem_type": "supervised",
        "task": "classification",
        "data_source": {"type": "csv", "path": str(dataset_path)},
        "target_column": "target",
        "validation": {"strategy": "out_of_sample", "test_size": 0.25},
        "model": {"algorithm": "logistic_regression"},
        "artifacts": {"root_path": str(tmp_path / "artifacts")},
    }), encoding="utf-8")

    result = CliRunner().invoke(app, ["--config", str(config_path)])

    assert result.exit_code == 0
    assert "Training completed" in result.stdout
