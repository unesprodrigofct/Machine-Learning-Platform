"""Load and validate the immutable artifact consumed by inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ml_platform.domain.errors import ArtifactError


@dataclass(frozen=True)
class ArtifactContext:
    pipeline: Any
    schema: dict[str, str]
    metadata: dict[str, Any]
    artifact_path: Path

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("run_id", self.artifact_path.name))


class ArtifactLoader:
    """Filesystem adapter for a versioned inference artifact."""

    REQUIRED_FILES = ("inference_pipeline.joblib", "schema.json", "metadata.json")

    def __init__(self, artifact_path: str | Path) -> None:
        self._artifact_path = Path(artifact_path)

    def load_pipeline(self) -> Any:
        path = self._artifact_path / "inference_pipeline.joblib"
        if not path.is_file():
            raise ArtifactError(f"Inference pipeline not found: {path.name}")
        try:
            return joblib.load(path)
        except Exception as error:
            raise ArtifactError("Inference pipeline could not be loaded.") from error

    def load_schema(self) -> dict[str, str]:
        schema = self._load_json("schema.json")
        if not isinstance(schema, dict) or not all(
            isinstance(name, str) and isinstance(dtype, str) for name, dtype in schema.items()
        ):
            raise ArtifactError("Artifact schema must map feature names to type names.")
        return schema

    def load_metadata(self) -> dict[str, Any]:
        metadata = self._load_json("metadata.json")
        if not isinstance(metadata, dict):
            raise ArtifactError("Artifact metadata must be a JSON object.")
        return metadata

    def load(self) -> ArtifactContext:
        if not self._artifact_path.is_dir():
            raise ArtifactError("Configured inference artifact directory was not found.")
        missing = [name for name in self.REQUIRED_FILES if not (self._artifact_path / name).is_file()]
        if missing:
            raise ArtifactError(f"Inference artifact is missing required files: {', '.join(missing)}")
        return ArtifactContext(
            pipeline=self.load_pipeline(),
            schema=self.load_schema(),
            metadata=self.load_metadata(),
            artifact_path=self._artifact_path,
        )

    def _load_json(self, filename: str) -> Any:
        path = self._artifact_path / filename
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"Artifact file {filename} is invalid.") from error
