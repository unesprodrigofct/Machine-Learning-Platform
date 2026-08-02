from pathlib import Path

import pytest

from ml_platform.domain.config import load_training_config
from ml_platform.domain.errors import ConfigurationError, TrainingError


def test_configuration_error_contains_machine_readable_code(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("problem_type: supervised\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"\[CONFIGURATION_ERROR\]") as error:
        load_training_config(config_path)

    assert "name" in str(error.value)


def test_training_error_preserves_operational_details() -> None:
    error = TrainingError("Training could not be completed.", details={"reason": "bad parameter"})

    assert str(error) == "[TRAINING_ERROR] Training could not be completed."
    assert error.details["reason"] == "bad parameter"
