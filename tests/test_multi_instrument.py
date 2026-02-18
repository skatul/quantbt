"""Tests for synchronized bar iteration, multi-instrument engine, and equity tracking."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from quantbt.broker.simulated import SimulatedBroker
from quantbt.core.engine import BacktestEngine
from quantbt.data.bar import Bar
from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument, equity
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.base import Strategy
from quantbt.strategy.multi_instrument import MultiInstrumentStrategy


# ---------- Mock DataFeed ----------

class MockMultiDataFeed(DataFeed):
    """DataFeed backed by in-memory bar lists per symbol."""

    def __init__(self, bars_by_symbol: dict[str, list[Bar]]) -> None:
        self._bars = bars_by_symbol

    def fetch(self, instrument: Instrument, start: datetime, end: datetime) -> pd.DataFrame:
        bars = self._bars.get(instrument.symbol, [])
        rows = []
        for b in bars:
            if start <= b.timestamp <= end:
                rows.append({
                    "open": b.open, "high": b.high, "low": b.low,
                    "close": b.close, "volume": b.volume,
                    "timestamp": b.timestamp,
                })
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).set_index("timestamp")
        return df


def _make_bars(symbol: str, prices: list[float], base_date="2024-01-01") -> list[Bar]:
    dates = pd.date_range(base_date, periods=len(prices), freq="D")
    return [
        Bar(
            timestamp=d.to_pydatetime(),
            open=p - 0.1, high=p + 0.5, low=p - 0.5,
            close=p, volume=1000.0, symbol=symbol,
        )
        for d, p in zip(dates, prices)
    ]


# ---------- Test iter_bars_sync ----------

class TestIterBarsSync:
    def test_merges_by_timestamp(self):
        bars_a = _make_bars("A", [10, 11, 12])
        bars_b = _make_bars("B", [20, 21, 22])
        feed = MockMultiDataFeed({"A": bars_a, "B": bars_b})
        inst_a = equity("A")
        inst_b = equity("B")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        result = list(feed.iter_bars_sync([inst_a, inst_b], start, end))
        assert len(result) == 3
        for ts, bars in result:
            assert "A" in bars
            assert "B" in bars

    def test_handles_missing_timestamps(self):
        bars_a = _make_bars("A", [10, 11, 12])
        # B only has 2 bars (2024-01-01, 2024-01-02)
        bars_b = _make_bars("B", [20, 21])
        feed = MockMultiDataFeed({"A": bars_a, "B": bars_b})
        inst_a = equity("A")
        inst_b = equity("B")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        result = list(feed.iter_bars_sync([inst_a, inst_b], start, end))
        assert len(result) == 3
        # Last timestamp only has A
        last_ts, last_bars = result[-1]
        assert "A" in last_bars
        assert "B" not in last_bars

    def test_chronological_order(self):
        bars_a = _make_bars("A", [10, 11, 12, 13])
        bars_b = _make_bars("B", [20, 21, 22, 23])
        feed = MockMultiDataFeed({"A": bars_a, "B": bars_b})
        inst_a = equity("A")
        inst_b = equity("B")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 4)

        result = list(feed.iter_bars_sync([inst_a, inst_b], start, end))
        timestamps = [ts for ts, _ in result]
        assert timestamps == sorted(timestamps)


# ---------- Test MultiInstrumentStrategy ----------

class SpreadTracker(MultiInstrumentStrategy):
    """Simple test strategy that tracks the spread between two symbols."""

    def __init__(self, broker, portfolio, sym_a, sym_b):
        super().__init__(broker, portfolio)
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.spreads: list[float] = []

    def on_bars(self, bars: dict[str, Bar]) -> None:
        if self.sym_a in bars and self.sym_b in bars:
            spread = bars[self.sym_a].close - bars[self.sym_b].close
            self.spreads.append(spread)


class TestMultiInstrumentEngine:
    def test_synchronized_mode(self):
        bars_a = _make_bars("A", [100, 101, 102, 103, 104])
        bars_b = _make_bars("B", [50, 51, 52, 53, 54])
        feed = MockMultiDataFeed({"A": bars_a, "B": bars_b})

        broker = SimulatedBroker(commission_rate=0.0)
        portfolio = Portfolio(initial_cash=100_000.0)
        strategy = SpreadTracker(broker, portfolio, "A", "B")

        inst_a = equity("A")
        inst_b = equity("B")
        engine = BacktestEngine(
            data_feed=feed, strategy=strategy, broker=broker,
            instruments=[inst_a, inst_b],
            start=datetime(2024, 1, 1), end=datetime(2024, 1, 5),
        )
        engine.run()

        assert len(strategy.spreads) == 5
        assert strategy.spreads[0] == pytest.approx(50.0)

    def test_equity_records_for_each_timestamp(self):
        bars_a = _make_bars("A", [100, 101, 102])
        bars_b = _make_bars("B", [200, 201, 202])
        feed = MockMultiDataFeed({"A": bars_a, "B": bars_b})

        broker = SimulatedBroker(commission_rate=0.0)
        portfolio = Portfolio(initial_cash=100_000.0)
        strategy = SpreadTracker(broker, portfolio, "A", "B")

        inst_a = equity("A")
        inst_b = equity("B")
        engine = BacktestEngine(
            data_feed=feed, strategy=strategy, broker=broker,
            instruments=[inst_a, inst_b],
            start=datetime(2024, 1, 1), end=datetime(2024, 1, 3),
        )
        engine.run()

        assert len(engine.performance.equity_curve) == 3


# ---------- Test backward compatibility ----------

class SimpleCounter(Strategy):
    """Single-instrument strategy that counts bars."""

    def __init__(self, broker, portfolio):
        super().__init__(broker, portfolio)
        self.count = 0

    def on_bar(self, bar: Bar) -> None:
        self.count += 1


class TestBackwardCompatibility:
    def test_single_instrument_still_works(self):
        bars = _make_bars("X", [10, 11, 12, 13, 14])
        feed = MockMultiDataFeed({"X": bars})

        broker = SimulatedBroker(commission_rate=0.0)
        portfolio = Portfolio(initial_cash=100_000.0)
        strategy = SimpleCounter(broker, portfolio)

        engine = BacktestEngine(
            data_feed=feed, strategy=strategy, broker=broker,
            instruments=[equity("X")],
            start=datetime(2024, 1, 1), end=datetime(2024, 1, 5),
        )
        engine.run()

        assert strategy.count == 5

    def test_default_on_bars_calls_on_bar(self):
        """Strategy.on_bars default should call on_bar for each bar."""
        bars_a = _make_bars("A", [10, 11])
        feed = MockMultiDataFeed({"A": bars_a})

        broker = SimulatedBroker(commission_rate=0.0)
        portfolio = Portfolio(initial_cash=100_000.0)
        strategy = SimpleCounter(broker, portfolio)

        # Even with a single instrument, default on_bars works
        bar = bars_a[0]
        strategy.on_bars({"A": bar})
        assert strategy.count == 1
