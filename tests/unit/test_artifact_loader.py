import json
from pathlib import Path

import joblib
import pytest

from ml_platform.adapters.artifact_stores.artifact_loader import ArtifactLoader
from ml_platform.domain.errors import ArtifactError


class DummyPipeline:
    pass


def _write_artifact(path: Path, *, metadata: object = None, schema: object = None) -> None:
    path.mkdir()
    joblib.dump(DummyPipeline(), path / "inference_pipeline.joblib")
    (path / "schema.json").write_text(json.dumps(schema or {"feature": "float64"}), encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps(metadata or {"run_id": "run-123"}), encoding="utf-8")


def test_loader_loads_complete_artifact_and_falls_back_to_directory_name(tmp_path: Path) -> None:
    artifact = tmp_path / "run-123"
    _write_artifact(artifact, metadata={})

    context = ArtifactLoader(artifact).load()

    assert context.schema == {"feature": "float64"}
    assert context.run_id == "run-123"
    assert context.artifact_path == artifact


@pytest.mark.parametrize("missing", ["inference_pipeline.joblib", "schema.json", "metadata.json"])
def test_loader_reports_missing_required_files(tmp_path: Path, missing: str) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    (artifact / missing).unlink()

    with pytest.raises(ArtifactError, match="missing required files"):
        ArtifactLoader(artifact).load()


def test_loader_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="directory was not found"):
        ArtifactLoader(tmp_path / "missing").load()


def test_loader_rejects_invalid_schema_and_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, schema=["feature"])
    with pytest.raises(ArtifactError, match="schema must map"):
        ArtifactLoader(artifact).load_schema()

    (artifact / "metadata.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ArtifactError, match="metadata must be"):
        ArtifactLoader(artifact).load_metadata()


def test_loader_reports_invalid_json_and_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    (artifact / "schema.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(ArtifactError, match="schema.json is invalid"):
        ArtifactLoader(artifact).load_schema()

    def fail_load(_: Path) -> object:
        raise RuntimeError("bad joblib")

    monkeypatch.setattr(joblib, "load", fail_load)
    with pytest.raises(ArtifactError, match="could not be loaded"):
        ArtifactLoader(artifact).load_pipeline()


def test_loader_reports_missing_pipeline_file(tmp_path: Path) -> None:
    loader = ArtifactLoader(tmp_path / "artifact")
    with pytest.raises(ArtifactError, match="pipeline not found"):
        loader.load_pipeline()
