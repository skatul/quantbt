from __future__ import annotations

from datetime import datetime

import pandas as pd

from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument


class CcxtDataFeed(DataFeed):
    """Fetches OHLCV data via the ccxt library.

    Requires ``pip install quantbt[crypto]`` for the ccxt dependency.
    """

    def __init__(self, exchange: str = "binance", timeframe: str = "1d") -> None:
        self.exchange_id = exchange
        self.timeframe = timeframe
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is None:
            try:
                import ccxt
            except ImportError as e:
                raise ImportError(
                    "ccxt is required for CcxtDataFeed. "
                    "Install with: pip install quantbt[crypto]"
                ) from e
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    def fetch(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        exchange = self._get_exchange()
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_ohlcv: list[list] = []
        while since < end_ms:
            ohlcv = exchange.fetch_ohlcv(
                instrument.symbol,
                timeframe=self.timeframe,
                since=since,
                limit=1000,
            )
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        if not all_ohlcv:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]
