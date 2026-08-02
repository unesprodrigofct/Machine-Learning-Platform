import pytest

from ml_platform.domain.config import TrainingConfig, resolve_metrics


def test_supervised_training_requires_target_column() -> None:
    with pytest.raises(ValueError, match="require target_column"):
        TrainingConfig.model_validate({
            "name": "missing-target",
            "problem_type": "supervised",
            "task": "classification",
            "data_source": {"type": "csv", "path": "data.csv"},
            "validation": {"strategy": "out_of_sample"},
            "model": {"algorithm": "logistic_regression"},
        })


def test_temporal_validation_requires_date_column() -> None:
    with pytest.raises(ValueError, match="requires date_column"):
        TrainingConfig.model_validate({
            "name": "temporal",
            "problem_type": "unsupervised",
            "task": "clustering",
            "data_source": {"type": "csv", "path": "data.csv"},
            "validation": {"strategy": "out_of_time_last_n_months", "validation_months": 3},
            "model": {"algorithm": "kmeans"},
        })


def test_metric_defaults_are_loaded_from_yaml() -> None:
    config = TrainingConfig.model_validate({
        "name": "default-metrics",
        "problem_type": "supervised",
        "task": "classification",
        "data_source": {"type": "csv", "path": "data.csv"},
        "target_column": "target",
        "validation": {"strategy": "out_of_sample"},
        "model": {"algorithm": "logistic_regression"},
    })

    assert "roc_auc" in resolve_metrics(config)
    assert "ks" in resolve_metrics(config)


def test_explicit_metrics_override_yaml_defaults() -> None:
    config = TrainingConfig.model_validate({
        "name": "custom-metrics",
        "problem_type": "supervised",
        "task": "classification",
        "data_source": {"type": "csv", "path": "data.csv"},
        "target_column": "target",
        "validation": {"strategy": "out_of_sample"},
        "model": {"algorithm": "logistic_regression"},
        "evaluation": {"metrics": ["accuracy"]},
    })

    assert resolve_metrics(config) == ["accuracy"]
