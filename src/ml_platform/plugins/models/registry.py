"""Resolve model plugins separately from the training orchestration."""

from ml_platform.domain.config import TrainingConfig
from ml_platform.domain.errors import PluginResolutionError
from ml_platform.plugins.models.sklearn_tabular import SklearnTabularModelPlugin


def resolve_model_plugin(config: TrainingConfig) -> SklearnTabularModelPlugin:
    plugin = SklearnTabularModelPlugin()
    if plugin.supports(config.task, config.model.algorithm):
        return plugin
    raise PluginResolutionError(f"No registered model plugin supports {config.task}/{config.model.algorithm}.")
