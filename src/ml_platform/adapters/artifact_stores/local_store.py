"""Local filesystem artifact-store adapter."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import sklearn
import yaml


def _package_version() -> str:
    try:
        return version("machine-learning-platform")
    except PackageNotFoundError:
        return "unknown"


class LocalArtifactStore:
    def __init__(self, root_path: str) -> None:
        self._root_path = Path(root_path)

    def save(self, *, pipeline: Any, config: dict[str, Any], metrics: dict[str, float], metadata: dict[str, Any], schema: dict[str, str]) -> tuple[str, str]:
        config_bytes = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{hashlib.sha256(config_bytes).hexdigest()[:8]}"
        run_path = self._root_path / run_id
        run_path.mkdir(parents=True, exist_ok=False)
        joblib.dump(pipeline, run_path / "inference_pipeline.joblib")
        (run_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_path / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        full_metadata = {
            **metadata,
            "run_id": run_id,
            "artifact_version": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "training_code_version": metadata.get("training_code_version", _package_version()),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
        }
        (run_path / "metadata.json").write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
        (run_path / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        return run_id, str(run_path)
