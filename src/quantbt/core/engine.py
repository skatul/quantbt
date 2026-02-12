from __future__ import annotations

from datetime import datetime

from quantbt.broker.base import Broker
from quantbt.broker.simulated import SimulatedBroker
from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument
from quantbt.orders.model import Fill, Order
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.base import Strategy


class BacktestEngine:
    """Main orchestrator. Feeds bars to strategy, coordinates broker and portfolio."""

    def __init__(
        self,
        data_feed: DataFeed,
        strategy: Strategy,
        broker: Broker,
        instruments: list[Instrument],
        start: datetime,
        end: datetime,
    ) -> None:
        self.data_feed = data_feed
        self.strategy = strategy
        self.broker = broker
        self.instruments = instruments
        self.start = start
        self.end = end

        # Wire up fill callback to update portfolio
        self.broker.on_fill(self._on_fill)

    def run(self) -> Portfolio:
        self.strategy.on_init()

        for instrument in self.instruments:
            for bar in self.data_feed.iter_bars(instrument, self.start, self.end):
                # If simulated broker, set current bar for fill pricing
                if isinstance(self.broker, SimulatedBroker):
                    self.broker.set_current_bar(bar)

                self.strategy.on_bar(bar)

                # If live broker, poll for ZMQ responses
                if hasattr(self.broker, "poll_responses"):
                    self.broker.poll_responses()

        return self.strategy.portfolio

    def _on_fill(self, order: Order, fill: Fill) -> None:
        self.strategy.portfolio.update_position(
            symbol=order.symbol,
            side=order.side.value,
            quantity=fill.fill_quantity,
            price=fill.fill_price,
            commission=fill.commission,
        )
        self.strategy.on_fill(order, fill)
