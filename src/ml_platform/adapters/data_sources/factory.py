"""Resolve configured data-source adapters."""

from ml_platform.adapters.data_sources.csv_source import CsvDataSource
from ml_platform.adapters.data_sources.database_source import DatabaseDataSource
from ml_platform.domain.config import DataSourceConfig
from ml_platform.domain.errors import ConfigurationError
from ml_platform.ports.data_source import DataSource


def create_data_source(config: DataSourceConfig) -> DataSource:
    if config.type == "csv":
        return CsvDataSource(config)
    if config.type == "database":
        return DatabaseDataSource(config)
    raise ConfigurationError(f"Unsupported data source type: {config.type}")
