"""CSV data-source adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ml_platform.domain.config import DataSourceConfig
from ml_platform.domain.errors import DataValidationError


class CsvDataSource:
    def __init__(self, config: DataSourceConfig) -> None:
        self._path = Path(config.path or "")
        self._delimiter = config.delimiter
        self._encoding = config.encoding

    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise DataValidationError(f"CSV source not found: {self._path}")
        dataframe = pd.read_csv(self._path, sep=self._delimiter, encoding=self._encoding)
        if dataframe.empty:
            raise DataValidationError("The CSV source contains no rows.")
        return dataframe

    def fingerprint(self) -> str:
        return hashlib.sha256(self._path.read_bytes()).hexdigest()
