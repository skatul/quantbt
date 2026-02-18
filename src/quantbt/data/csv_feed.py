from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument


class CsvDataFeed(DataFeed):
    """Loads OHLCV data from a CSV file.

    Expects columns: open, high, low, close, volume (case-insensitive).
    Index or first date-like column is used as the datetime index.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def fetch(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.file_path, parse_dates=True, index_col=0)
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"CSV missing required columns. Found: {list(df.columns)}, "
                f"need: {sorted(required)}"
            )
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask, list(required)]
