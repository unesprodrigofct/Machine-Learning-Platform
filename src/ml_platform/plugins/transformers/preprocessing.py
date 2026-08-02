"""Construct the fitted preprocessing section of a training pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_platform.domain.config import TrainingConfig
from ml_platform.domain.errors import DataValidationError


def resolve_feature_columns(dataframe: pd.DataFrame, config: TrainingConfig) -> list[str]:
    excluded = set(config.features.exclude) | set(config.features.identifier_columns)
    if config.target_column:
        excluded.add(config.target_column)
    if config.validation.date_column:
        excluded.add(config.validation.date_column)
    columns = config.features.include or [column for column in dataframe.columns if column not in excluded]
    missing = set(columns) - set(dataframe.columns)
    if missing:
        raise DataValidationError(f"Configured feature columns are missing: {sorted(missing)}")
    if not columns:
        raise DataValidationError("No feature columns remain after applying the feature configuration.")
    return columns


def build_preprocessor(dataframe: pd.DataFrame, config: TrainingConfig, feature_columns: list[str]) -> ColumnTransformer:
    configured_numeric = config.features.numeric_columns
    configured_categorical = config.features.categorical_columns
    numeric_columns = configured_numeric or dataframe[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical_columns = configured_categorical or [column for column in feature_columns if column not in numeric_columns]
    unknown = (set(numeric_columns) | set(categorical_columns)) - set(feature_columns)
    if unknown:
        raise DataValidationError(f"Preprocessing columns are not selected features: {sorted(unknown)}")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy=config.preprocessing.numeric_imputer))]
        if config.preprocessing.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy=config.preprocessing.categorical_imputer, fill_value=config.preprocessing.categorical_fill_value))
        ]
        if config.preprocessing.one_hot_encode:
            categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_columns))
    if not transformers:
        raise DataValidationError("No compatible numeric or categorical feature columns were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
