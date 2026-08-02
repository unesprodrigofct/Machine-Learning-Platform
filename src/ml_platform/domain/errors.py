"""Domain errors exposed consistently across CLI and API boundaries."""


class PlatformError(Exception):
    code = "PLATFORM_ERROR"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

class ConfigurationError(PlatformError):
    code = "CONFIGURATION_ERROR"

class DataValidationError(PlatformError):
    code = "DATA_VALIDATION_ERROR"

class PluginResolutionError(PlatformError):
    code = "PLUGIN_RESOLUTION_ERROR"

class TrainingError(PlatformError):
    code = "TRAINING_ERROR"

class ArtifactError(PlatformError):
    code = "ARTIFACT_ERROR"
