"""Training application service: the framework-independent orchestration flow."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ml_platform.adapters.artifact_stores.local_store import LocalArtifactStore
from ml_platform.adapters.data_sources.factory import create_data_source
from ml_platform.domain.config import TrainingConfig, resolve_metrics
from ml_platform.domain.errors import DataValidationError, PlatformError, TrainingError
from ml_platform.domain.models import TrainingResult
from ml_platform.plugins.estimators.registry import create_estimator
from ml_platform.plugins.evaluators.metrics import evaluate
from ml_platform.plugins.transformers.preprocessing import build_preprocessor, resolve_feature_columns
from ml_platform.plugins.validation.strategies import create_split
from ml_platform.plugins.models.registry import resolve_model_plugin


class TrainModelUseCase:
    def execute(self, config: TrainingConfig) -> TrainingResult:
        model_plugin = resolve_model_plugin(config)
        model_plugin.validate(config)
        # Resolve the effective metric list before loading/fitting data. This
        # makes YAML defaults part of configuration validation and prevents a
        # run from doing work before an invalid defaults file is reported.
        requested_metrics = resolve_metrics(config)
        source = create_data_source(config.data_source)
        dataframe = source.load()
        self._validate_dataset(dataframe.columns.tolist(), config)
        split = create_split(dataframe, config)

        feature_columns = resolve_feature_columns(split.train, config)
        preprocessor = build_preprocessor(split.train, config, feature_columns)
        estimator = create_estimator(config.model.algorithm, config.task, config.model.parameters, config.random_seed)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        train_target = split.train[config.target_column] if config.target_column else None
        trained_pipeline = self._fit(pipeline, split.train[feature_columns], train_target, config)

        validation_target = split.validation[config.target_column] if config.target_column else None
        metrics = evaluate(
            trained_pipeline,
            split.validation[feature_columns],
            validation_target,
            config.task,
            requested_metrics,
        )
        schema = {column: str(split.train[column].dtype) for column in feature_columns}
        effective_config = config.model_dump(mode="json")
        effective_config["evaluation"]["metrics"] = requested_metrics
        metadata: dict[str, Any] = {
            "training_name": config.name,
            "model_type": model_plugin.model_type,
            "problem_type": config.problem_type,
            "task": config.task,
            "algorithm": config.model.algorithm,
            "random_seed": config.random_seed,
            "data_fingerprint": source.fingerprint(),
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
        }
        run_id, artifact_path = LocalArtifactStore(config.artifacts.root_path).save(
            pipeline=trained_pipeline,
            config=effective_config,
            metrics=metrics,
            metadata=metadata,
            schema=schema,
        )
        return TrainingResult(run_id=run_id, artifact_path=artifact_path, metrics=metrics)

    @staticmethod
    def _validate_dataset(columns: list[str], config: TrainingConfig) -> None:
        if config.target_column and config.target_column not in columns:
            raise DataValidationError(f"Target column is missing from the dataset: {config.target_column!r}")

    @staticmethod
    def _fit(pipeline: Pipeline, features: Any, target: Any, config: TrainingConfig) -> Pipeline:
        search = config.hyperparameter_search
        if search.enabled:
            if config.problem_type != "supervised":
                raise DataValidationError("Hyperparameter search is currently supported for supervised tasks only.")
            parameter_space = {
                key if key.startswith("model__") else f"model__{key}": value
                for key, value in search.parameters.items()
            }
            optimizer = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameter_space,
                n_iter=search.n_iter,
                cv=search.cv,
                scoring=search.scoring,
                random_state=config.random_seed,
                n_jobs=-1,
            )
            optimizer.fit(features, target)
            return optimizer.best_estimator_
        return pipeline.fit(features, target) if config.problem_type == "supervised" else pipeline.fit(features)


def train(config: TrainingConfig) -> TrainingResult:
    try:
        return TrainModelUseCase().execute(config)
    except PlatformError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError("Training could not be completed.", details={"reason": str(error)}) from error
