"""Relational database adapter with guarded read-only SQL execution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from ml_platform.domain.config import DataSourceConfig
from ml_platform.domain.errors import DataValidationError


class DatabaseDataSource:
    def __init__(self, config: DataSourceConfig) -> None:
        self._config = config

    def _query(self) -> str:
        query = self._config.query
        if self._config.query_file:
            try:
                query = Path(self._config.query_file).read_text(encoding="utf-8")
            except FileNotFoundError as error:
                raise DataValidationError(f"SQL query file not found: {self._config.query_file}") from error
        normalized = (query or "").strip().lower()
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise DataValidationError("Database queries must be read-only SELECT or WITH statements.")
        return query or ""

    def _connection_url(self) -> str:
        value = os.getenv(self._config.connection_env or "")
        if not value:
            raise DataValidationError(f"Environment variable {self._config.connection_env!r} is not set.")
        return value

    def load(self) -> pd.DataFrame:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as error:
            raise DataValidationError("Database support requires `pip install .[database]`.") from error
        with create_engine(self._connection_url()).connect() as connection:
            dataframe = pd.read_sql_query(text(self._query()), connection)
        if dataframe.empty:
            raise DataValidationError("The database query returned no rows.")
        return dataframe

    def fingerprint(self) -> str:
        return hashlib.sha256(self._query().encode("utf-8")).hexdigest()
