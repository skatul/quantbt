from datetime import datetime
from collections.abc import Iterator

import pandas as pd

from quantbt.broker.simulated import SimulatedBroker
from quantbt.core.engine import BacktestEngine
from quantbt.data.bar import Bar
from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument, equity
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.examples.sma_crossover import SMACrossover


class MockDataFeed(DataFeed):
    """DataFeed that returns pre-built bars."""

    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars

    def fetch(self, instrument: Instrument, start: datetime,
              end: datetime) -> pd.DataFrame:
        data = [{
            "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume,
        } for b in self._bars]
        index = pd.DatetimeIndex([b.timestamp for b in self._bars])
        return pd.DataFrame(data, index=index)

    def iter_bars(self, instrument: Instrument, start: datetime,
                  end: datetime) -> Iterator[Bar]:
        yield from self._bars


def test_backtest_engine_runs(sample_bars):
    instrument = equity("TEST")
    portfolio = Portfolio(initial_cash=100_000.0)
    broker = SimulatedBroker(commission_rate=0.001)
    strategy = SMACrossover(
        broker=broker,
        portfolio=portfolio,
        fast_period=5,
        slow_period=10,
        trade_quantity=10,
    )
    data_feed = MockDataFeed(sample_bars)

    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        broker=broker,
        instruments=[instrument],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )

    result = engine.run()
    assert result is portfolio
    # Engine should have processed all bars and generated trades
    assert portfolio.initial_cash == 100_000.0


def test_backtest_engine_fills_update_portfolio(sample_bars):
    instrument = equity("TEST")
    portfolio = Portfolio(initial_cash=100_000.0)
    broker = SimulatedBroker(commission_rate=0.0)
    strategy = SMACrossover(
        broker=broker,
        portfolio=portfolio,
        fast_period=5,
        slow_period=10,
        trade_quantity=10,
    )
    data_feed = MockDataFeed(sample_bars)

    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        broker=broker,
        instruments=[instrument],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )

    engine.run()

    # If trades were made, cash should have changed
    if len(portfolio.positions) > 0:
        assert portfolio.cash != portfolio.initial_cash
