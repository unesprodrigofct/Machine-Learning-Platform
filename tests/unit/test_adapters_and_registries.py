from pathlib import Path

import pytest

from ml_platform.adapters.data_sources.database_source import DatabaseDataSource
from ml_platform.adapters.data_sources.factory import create_data_source
from ml_platform.domain.config import DataSourceConfig, TrainingConfig
from ml_platform.domain.errors import ConfigurationError, DataValidationError, PluginResolutionError
from ml_platform.plugins.estimators.registry import create_estimator
from ml_platform.plugins.models.registry import resolve_model_plugin


def test_database_source_reads_query_file_and_fingerprints_it(tmp_path: Path) -> None:
    query_file = tmp_path / "query.sql"
    query_file.write_text("WITH orders AS (SELECT 1) SELECT * FROM orders", encoding="utf-8")
    source = DatabaseDataSource(DataSourceConfig(type="database", connection_env="DATABASE_URL", query_file=str(query_file)))

    assert source._query().startswith("WITH")
    assert len(source.fingerprint()) == 64


def test_database_source_rejects_missing_query_file_and_connection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = DatabaseDataSource(DataSourceConfig(type="database", connection_env="DATABASE_URL", query_file=str(tmp_path / "missing.sql")))
    with pytest.raises(DataValidationError, match="query file not found"):
        source._query()

    source = DatabaseDataSource(DataSourceConfig(type="database", connection_env="DATABASE_URL", query="SELECT 1"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DataValidationError, match="is not set"):
        source._connection_url()


def test_data_source_factory_resolves_csv_and_database() -> None:
    csv_source = create_data_source(DataSourceConfig(type="csv", path="data.csv"))
    database_source = create_data_source(DataSourceConfig(type="database", connection_env="DATABASE_URL", query="SELECT 1"))

    assert csv_source.__class__.__name__ == "CsvDataSource"
    assert database_source.__class__.__name__ == "DatabaseDataSource"


def test_data_source_factory_rejects_unknown_type() -> None:
    config = object.__new__(DataSourceConfig)
    object.__setattr__(config, "type", "unknown")
    with pytest.raises(ConfigurationError, match="Unsupported data source"):
        create_data_source(config)


@pytest.mark.parametrize(
    ("algorithm", "task"),
    [
        ("logistic_regression", "classification"),
        ("linear_regression", "regression"),
        ("random_forest_regressor", "regression"),
        ("kmeans", "clustering"),
        ("isolation_forest", "anomaly_detection"),
    ],
)
def test_estimator_registry_resolves_supported_algorithms(algorithm: str, task: str) -> None:
    estimator = create_estimator(algorithm, task, {}, 7)
    if algorithm in {"random_forest_regressor", "kmeans", "isolation_forest"}:
        assert estimator.random_state == 7


def test_model_plugin_supports_valid_configuration() -> None:
    config = TrainingConfig.model_validate({
        "name": "plugin",
        "problem_type": "supervised",
        "task": "classification",
        "data_source": {"type": "csv", "path": "data.csv"},
        "target_column": "target",
        "validation": {"strategy": "out_of_sample"},
        "model": {"algorithm": "logistic_regression"},
    })
    plugin = resolve_model_plugin(config)

    assert plugin.capabilities()["model_type"] == "sklearn_tabular"
    plugin.validate(config)


def test_estimator_registry_reports_invalid_resolution() -> None:
    with pytest.raises(PluginResolutionError, match="not compatible"):
        create_estimator("linear_regression", "classification", {}, 42)
