"""Port for tabular data sources."""

from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    def load(self) -> pd.DataFrame:
        ...

    def fingerprint(self) -> str:
        ...
