from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument


class YahooFinanceDataFeed(DataFeed):
    def fetch(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        ticker = yf.Ticker(instrument.symbol)
        df = ticker.history(start=start, end=end)
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "timestamp"
        return df
