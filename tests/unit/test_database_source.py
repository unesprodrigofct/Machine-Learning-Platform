import pytest

from ml_platform.adapters.data_sources.database_source import DatabaseDataSource
from ml_platform.domain.config import DataSourceConfig
from ml_platform.domain.errors import DataValidationError


def test_database_source_accepts_read_only_query() -> None:
    source = DatabaseDataSource(DataSourceConfig(type="database", connection_env="DATABASE_URL", query="SELECT * FROM orders"))

    assert source._query() == "SELECT * FROM orders"


def test_database_source_rejects_write_query() -> None:
    source = DatabaseDataSource(DataSourceConfig(type="database", connection_env="DATABASE_URL", query="DELETE FROM orders"))

    with pytest.raises(DataValidationError, match="read-only"):
        source._query()
