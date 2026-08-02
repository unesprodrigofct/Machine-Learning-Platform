"""Validated, serializable configuration models for training runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic import ValidationError

from ml_platform.domain.errors import ConfigurationError


class DataSourceConfig(BaseModel):
    type: Literal["csv", "database"]
    path: str | None = None
    delimiter: str = ","
    encoding: str = "utf-8"
    connection_env: str | None = None
    query: str | None = None
    query_file: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "DataSourceConfig":
        if self.type == "csv" and not self.path:
            raise ValueError("A CSV data source requires 'path'.")
        if self.type == "database":
            if not self.connection_env:
                raise ValueError("A database data source requires 'connection_env'.")
            if not self.query and not self.query_file:
                raise ValueError("A database data source requires 'query' or 'query_file'.")
            if self.query and self.query_file:
                raise ValueError("Provide either 'query' or 'query_file', not both.")
        return self


class ValidationConfig(BaseModel):
    strategy: Literal["out_of_sample", "out_of_time_range", "out_of_time_last_n_months"]
    test_size: float = Field(default=0.30, gt=0, lt=1)
    stratify: bool = True
    date_column: str | None = None
    validation_start_date: date | None = None
    validation_end_date: date | None = None
    validation_months: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_strategy(self) -> "ValidationConfig":
        if self.strategy == "out_of_time_range":
            if not all([self.date_column, self.validation_start_date, self.validation_end_date]):
                raise ValueError("out_of_time_range requires date_column, validation_start_date, and validation_end_date.")
            if self.validation_start_date > self.validation_end_date:
                raise ValueError("validation_start_date must be on or before validation_end_date.")
        if self.strategy == "out_of_time_last_n_months" and not all([self.date_column, self.validation_months]):
            raise ValueError("out_of_time_last_n_months requires date_column and validation_months.")
        return self


class FeatureConfig(BaseModel):
    include: list[str] | None = None
    exclude: list[str] = Field(default_factory=list)
    identifier_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None


class PreprocessingConfig(BaseModel):
    numeric_imputer: Literal["mean", "median"] = "median"
    scale_numeric: bool = True
    categorical_imputer: Literal["most_frequent", "constant"] = "most_frequent"
    categorical_fill_value: str = "missing"
    one_hot_encode: bool = True


class ModelConfig(BaseModel):
    algorithm: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class HyperparameterSearchConfig(BaseModel):
    enabled: bool = False
    n_iter: int = Field(default=10, gt=0)
    cv: int = Field(default=3, gt=1)
    scoring: str | None = None
    parameters: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_enabled_search(self) -> "HyperparameterSearchConfig":
        if self.enabled and not self.parameters:
            raise ValueError("Enabled hyperparameter_search requires a parameter search space.")
        return self


class EvaluationConfig(BaseModel):
    metrics: list[str] | None = None
    metric_defaults_path: str = "configs/metrics/defaults.yaml"
    selection_metric: str | None = None


class ArtifactConfig(BaseModel):
    root_path: str = "artifacts"


class TrainingConfig(BaseModel):
    name: str
    problem_type: Literal["supervised", "unsupervised"]
    task: Literal["classification", "regression", "clustering", "anomaly_detection"]
    data_source: DataSourceConfig
    target_column: str | None = None
    random_seed: int = 42
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    validation: ValidationConfig
    model: ModelConfig
    hyperparameter_search: HyperparameterSearchConfig = Field(default_factory=HyperparameterSearchConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)

    @model_validator(mode="after")
    def validate_problem(self) -> "TrainingConfig":
        supervised_tasks = {"classification", "regression"}
        unsupervised_tasks = {"clustering", "anomaly_detection"}
        if self.problem_type == "supervised" and self.task not in supervised_tasks:
            raise ValueError("Supervised problems must use classification or regression tasks.")
        if self.problem_type == "unsupervised" and self.task not in unsupervised_tasks:
            raise ValueError("Unsupervised problems must use clustering or anomaly_detection tasks.")
        if self.problem_type == "supervised" and not self.target_column:
            raise ValueError("Supervised problems require target_column.")
        if self.problem_type == "unsupervised" and self.target_column:
            raise ValueError("Unsupervised problems must not define target_column.")
        return self


def load_training_config(path: Path) -> TrainingConfig:
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise ConfigurationError("The configuration file must contain a YAML object.")
        return TrainingConfig.model_validate(raw_config)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML syntax in {path}: {error}") from error
    except ValidationError as error:
        fields = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        )
        raise ConfigurationError(f"Invalid training configuration: {fields}") from error


def resolve_metrics(config: TrainingConfig) -> list[str]:
    if config.evaluation.metrics is not None:
        return config.evaluation.metrics
    defaults_path = Path(config.evaluation.metric_defaults_path)
    try:
        defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Metric defaults file not found: {defaults_path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid metric defaults YAML: {error}") from error
    metrics = defaults.get(config.task) if isinstance(defaults, dict) else None
    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics):
        raise ConfigurationError(f"Metric defaults do not define a valid list for task: {config.task}")
    return metrics
