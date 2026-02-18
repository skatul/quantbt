from __future__ import annotations

from quantbt.broker.base import Broker
from quantbt.data.bar import Bar
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.multi_instrument import MultiInstrumentStrategy


class SpreadArbitrageStrategy(MultiInstrumentStrategy):
    """Cross-venue price discrepancy trading.

    Trades the same asset listed on two different venues (symbols).
    When the price difference exceeds a threshold, buys on the cheaper
    venue and sells on the more expensive one.
    """

    def __init__(
        self,
        broker: Broker,
        portfolio: Portfolio,
        venue_a: str,
        venue_b: str,
        threshold: float = 1.0,
        trade_quantity: float = 100.0,
    ) -> None:
        super().__init__(broker, portfolio)
        self.venue_a = venue_a
        self.venue_b = venue_b
        self.threshold = threshold
        self.trade_quantity = trade_quantity
        self._position: int = 0  # 0=flat, 1=long A / short B, -1=short A / long B

    def on_bars(self, bars: dict[str, Bar]) -> None:
        if self.venue_a not in bars or self.venue_b not in bars:
            return

        price_a = bars[self.venue_a].close
        price_b = bars[self.venue_b].close
        diff = price_a - price_b

        if self._position == 0:
            if diff > self.threshold:
                # A is expensive, B is cheap: sell A, buy B
                self.submit_order(self.venue_a, "sell", self.trade_quantity)
                self.submit_order(self.venue_b, "buy", self.trade_quantity)
                self._position = -1
            elif diff < -self.threshold:
                # B is expensive, A is cheap: buy A, sell B
                self.submit_order(self.venue_a, "buy", self.trade_quantity)
                self.submit_order(self.venue_b, "sell", self.trade_quantity)
                self._position = 1
        elif self._position == 1:
            # Close when spread reverts: A was cheap, now it's not
            if diff >= 0:
                self.submit_order(self.venue_a, "sell", self.trade_quantity)
                self.submit_order(self.venue_b, "buy", self.trade_quantity)
                self._position = 0
        elif self._position == -1:
            # Close when spread reverts: B was cheap, now it's not
            if diff <= 0:
                self.submit_order(self.venue_a, "buy", self.trade_quantity)
                self.submit_order(self.venue_b, "sell", self.trade_quantity)
                self._position = 0
