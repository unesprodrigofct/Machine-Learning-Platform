"""Port for persisting a completed training run."""

from typing import Any, Protocol


class ArtifactStore(Protocol):
    def save(self, *, pipeline: Any, config: dict[str, Any], metrics: dict[str, float], metadata: dict[str, Any], schema: dict[str, str]) -> tuple[str, str]:
        ...
