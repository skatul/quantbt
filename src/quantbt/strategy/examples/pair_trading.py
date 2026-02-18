from __future__ import annotations

from collections import deque
import math

from quantbt.broker.base import Broker
from quantbt.data.bar import Bar
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.multi_instrument import MultiInstrumentStrategy


class PairTradingStrategy(MultiInstrumentStrategy):
    """Statistical arbitrage on the z-score of the spread between two instruments.

    Goes long leg_a / short leg_b when z-score < -entry_z,
    goes short leg_a / long leg_b when z-score > +entry_z,
    exits when z-score crosses back through exit_z.
    """

    def __init__(
        self,
        broker: Broker,
        portfolio: Portfolio,
        leg_a: str,
        leg_b: str,
        lookback: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        trade_quantity: float = 100.0,
    ) -> None:
        super().__init__(broker, portfolio)
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.trade_quantity = trade_quantity
        self._spreads: deque[float] = deque(maxlen=lookback)
        self._position: int = 0  # 0=flat, 1=long spread, -1=short spread

    def on_bars(self, bars: dict[str, Bar]) -> None:
        if self.leg_a not in bars or self.leg_b not in bars:
            return

        spread = bars[self.leg_a].close - bars[self.leg_b].close
        self._spreads.append(spread)

        if len(self._spreads) < self.lookback:
            return

        mean = sum(self._spreads) / len(self._spreads)
        variance = sum((s - mean) ** 2 for s in self._spreads) / len(self._spreads)
        std = math.sqrt(variance)
        if std == 0:
            return
        z = (spread - mean) / std

        if self._position == 0:
            if z < -self.entry_z:
                # Long spread: buy A, sell B
                self.submit_order(self.leg_a, "buy", self.trade_quantity)
                self.submit_order(self.leg_b, "sell", self.trade_quantity)
                self._position = 1
            elif z > self.entry_z:
                # Short spread: sell A, buy B
                self.submit_order(self.leg_a, "sell", self.trade_quantity)
                self.submit_order(self.leg_b, "buy", self.trade_quantity)
                self._position = -1
        elif self._position == 1:
            if z > -self.exit_z:
                # Exit long spread
                self.submit_order(self.leg_a, "sell", self.trade_quantity)
                self.submit_order(self.leg_b, "buy", self.trade_quantity)
                self._position = 0
        elif self._position == -1:
            if z < self.exit_z:
                # Exit short spread
                self.submit_order(self.leg_a, "buy", self.trade_quantity)
                self.submit_order(self.leg_b, "sell", self.trade_quantity)
                self._position = 0
