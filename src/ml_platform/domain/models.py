"""Shared domain models passed across platform layers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    artifact_path: str
    metrics: dict[str, float]
