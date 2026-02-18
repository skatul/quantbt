from __future__ import annotations

import logging
from datetime import datetime

from quantbt.broker.base import Broker
from quantbt.broker.live import LiveBroker
from quantbt.broker.simulated import SimulatedBroker
from quantbt.data.base import DataFeed
from quantbt.data.bar import Bar
from quantbt.instrument.model import Instrument
from quantbt.orders.model import Fill, Order
from quantbt.portfolio.metrics import PerformanceTracker
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.base import Strategy

logger = logging.getLogger(__name__)


def _has_on_bars_override(strategy: Strategy) -> bool:
    """Check if the strategy overrides on_bars (i.e. is a multi-instrument strategy)."""
    return type(strategy).on_bars is not Strategy.on_bars


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
        self.performance = PerformanceTracker(
            initial_cash=strategy.portfolio.initial_cash
        )

        # Wire up fill callback to update portfolio
        self.broker.on_fill(self._on_fill)

    def run(self) -> Portfolio:
        self.strategy.on_init()
        logger.info("Starting backtest")

        if _has_on_bars_override(self.strategy) and len(self.instruments) > 1:
            self._run_synchronized()
        else:
            self._run_sequential()

        logger.info("Backtest complete. %s", self.performance.summary())
        return self.strategy.portfolio

    def _run_sequential(self) -> None:
        """Original sequential mode: process each instrument independently."""
        for instrument in self.instruments:
            for bar in self.data_feed.iter_bars(instrument, self.start, self.end):
                self._process_bar(bar)
                portfolio = self.strategy.portfolio
                market_prices = {bar.symbol: bar.close}
                mtm_equity = portfolio.mark_to_market(market_prices)
                positions_value = mtm_equity - portfolio.cash
                self.performance.record(
                    timestamp=bar.timestamp,
                    equity=mtm_equity,
                    cash=portfolio.cash,
                    positions_value=positions_value,
                )

    def _run_synchronized(self) -> None:
        """Synchronized mode: time-aligned bars across instruments."""
        for ts, bars in self.data_feed.iter_bars_sync(
            self.instruments, self.start, self.end
        ):
            # Update broker with all current bars
            if isinstance(self.broker, SimulatedBroker):
                for bar in bars.values():
                    self.broker.set_current_bar(bar)
                self.broker._current_bars = dict(bars)
            elif isinstance(self.broker, LiveBroker):
                for bar in bars.values():
                    self.broker.set_current_bar(bar)

            self.strategy.on_bars(bars)

            if isinstance(self.broker, LiveBroker):
                self.broker.poll_responses()

            # Record equity using all current market prices
            portfolio = self.strategy.portfolio
            market_prices = {sym: bar.close for sym, bar in bars.items()}
            mtm_equity = portfolio.mark_to_market(market_prices)
            positions_value = mtm_equity - portfolio.cash
            self.performance.record(
                timestamp=ts,
                equity=mtm_equity,
                cash=portfolio.cash,
                positions_value=positions_value,
            )

    def _process_bar(self, bar: Bar) -> None:
        """Set current bar and call strategy.on_bar."""
        if isinstance(self.broker, SimulatedBroker):
            self.broker.set_current_bar(bar)
        elif isinstance(self.broker, LiveBroker):
            self.broker.set_current_bar(bar)

        self.strategy.on_bar(bar)

        if isinstance(self.broker, LiveBroker):
            self.broker.poll_responses()

    def _on_fill(self, order: Order, fill: Fill) -> None:
        self.strategy.portfolio.update_position(
            symbol=order.symbol,
            side=order.side.value,
            quantity=fill.fill_quantity,
            price=fill.fill_price,
            commission=fill.commission,
        )
        self.strategy.on_fill(order, fill)
        logger.debug("Fill: %s %s %.0f @ %.2f",
                      order.side.value, order.symbol,
                      fill.fill_quantity, fill.fill_price)
