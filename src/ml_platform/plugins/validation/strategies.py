"""Deterministic validation-strategy plugins."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_platform.domain.config import TrainingConfig
from ml_platform.domain.errors import DataValidationError
from ml_platform.domain.models import DatasetSplit


def create_split(dataframe: pd.DataFrame, config: TrainingConfig) -> DatasetSplit:
    strategy = config.validation.strategy
    if strategy == "out_of_sample":
        return _out_of_sample(dataframe, config)
    if strategy == "out_of_time_range":
        return _out_of_time_range(dataframe, config)
    if strategy == "out_of_time_last_n_months":
        return _out_of_time_last_n_months(dataframe, config)
    raise DataValidationError(f"Unsupported validation strategy: {strategy}")


def _out_of_sample(dataframe: pd.DataFrame, config: TrainingConfig) -> DatasetSplit:
    stratify = None
    if config.task == "classification" and config.validation.stratify:
        assert config.target_column is not None
        stratify = dataframe[config.target_column]
    try:
        train, validation = train_test_split(
            dataframe,
            test_size=config.validation.test_size,
            random_state=config.random_seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as error:
        raise DataValidationError(f"Unable to create out-of-sample split: {error}") from error
    return DatasetSplit(train=train.copy(), validation=validation.copy())


def _date_series(dataframe: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in dataframe.columns:
        raise DataValidationError(f"Configured date column is missing: {column!r}")
    parsed = pd.to_datetime(dataframe[column], errors="coerce", utc=False)
    if parsed.isna().any():
        raise DataValidationError(f"Date column {column!r} contains null or unparseable values.")
    return parsed


def _out_of_time_range(dataframe: pd.DataFrame, config: TrainingConfig) -> DatasetSplit:
    dates = _date_series(dataframe, config.validation.date_column)
    start = pd.Timestamp(config.validation.validation_start_date.isoformat())
    end = pd.Timestamp(config.validation.validation_end_date.isoformat()) + pd.DateOffset(days=1)
    train = dataframe.loc[dates < start]
    validation = dataframe.loc[(dates >= start) & (dates < end)]
    return _validate_time_split(train, validation, "configured out-of-time range")


def _out_of_time_last_n_months(dataframe: pd.DataFrame, config: TrainingConfig) -> DatasetSplit:
    dates = _date_series(dataframe, config.validation.date_column)
    assert config.validation.validation_months is not None
    start = dates.max() - pd.DateOffset(months=config.validation.validation_months)
    train = dataframe.loc[dates < start]
    validation = dataframe.loc[dates >= start]
    return _validate_time_split(train, validation, f"last {config.validation.validation_months} months")


def _validate_time_split(train: pd.DataFrame, validation: pd.DataFrame, description: str) -> DatasetSplit:
    if train.empty or validation.empty:
        raise DataValidationError(f"Insufficient rows for {description}: both train and validation sets require data.")
    return DatasetSplit(train=train.copy(), validation=validation.copy())
