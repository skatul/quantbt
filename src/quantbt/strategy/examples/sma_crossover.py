from __future__ import annotations

from collections import deque

import pandas as pd
import numpy as np

from quantbt.broker.base import Broker
from quantbt.data.bar import Bar
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.base import Strategy
from quantbt.strategy.vectorized import VectorizedStrategy


class SMACrossover(Strategy):
    """Event-driven SMA crossover strategy."""

    def __init__(
        self,
        broker: Broker,
        portfolio: Portfolio,
        fast_period: int = 10,
        slow_period: int = 30,
        trade_quantity: float = 100.0,
    ) -> None:
        super().__init__(broker, portfolio)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trade_quantity = trade_quantity
        self._prices: deque[float] = deque(maxlen=slow_period)
        self._in_position = False

    def on_bar(self, bar: Bar) -> None:
        self._prices.append(bar.close)

        if len(self._prices) < self.slow_period:
            return

        prices = list(self._prices)
        fast_sma = sum(prices[-self.fast_period:]) / self.fast_period
        slow_sma = sum(prices) / self.slow_period

        if fast_sma > slow_sma and not self._in_position:
            self.submit_order(bar.symbol, "buy", self.trade_quantity)
            self._in_position = True
        elif fast_sma < slow_sma and self._in_position:
            self.submit_order(bar.symbol, "sell", self.trade_quantity)
            self._in_position = False


class SMACrossoverVectorized(VectorizedStrategy):
    """Vectorized SMA crossover strategy."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["fast_sma"] = df["close"].rolling(window=self.fast_period).mean()
        df["slow_sma"] = df["close"].rolling(window=self.slow_period).mean()

        df["signal"] = 0.0
        df.loc[df["fast_sma"] > df["slow_sma"], "signal"] = 1.0
        df.loc[df["fast_sma"] < df["slow_sma"], "signal"] = -1.0

        # Only signal on crossover (change in signal)
        df["signal"] = df["signal"].diff().apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        )
        return df
