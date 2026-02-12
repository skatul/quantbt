from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

import pandas as pd

from quantbt.data.bar import Bar
from quantbt.instrument.model import Instrument


class DataFeed(ABC):
    @abstractmethod
    def fetch(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Returns DataFrame with columns: open, high, low, close, volume, indexed by datetime."""
        ...

    def iter_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        df = self.fetch(instrument, start, end)
        for ts, row in df.iterrows():
            yield Bar(
                timestamp=ts.to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                symbol=instrument.symbol,
            )
