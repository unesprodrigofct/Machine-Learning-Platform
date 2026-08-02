"""Framework- and modality-neutral dataset contract."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Dataset:
    data: Any
    modality: Literal["tabular", "text", "image", "custom"]
    schema: dict[str, str]
    fingerprint: str
