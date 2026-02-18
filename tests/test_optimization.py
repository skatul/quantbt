"""Tests for parameter optimization: grid search, random search, walk-forward."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from quantbt.broker.simulated import SimulatedBroker
from quantbt.core.engine import BacktestEngine
from quantbt.data.bar import Bar
from quantbt.data.base import DataFeed
from quantbt.instrument.model import Instrument, equity
from quantbt.optimization.optimizer import Optimizer
from quantbt.optimization.parameter import ParameterSpace
from quantbt.optimization.walk_forward import WalkForwardOptimizer
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.base import Strategy


# ---------- Mock infrastructure ----------

class MockDataFeed(DataFeed):
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars

    def fetch(self, instrument: Instrument, start: datetime, end: datetime) -> pd.DataFrame:
        rows = []
        for b in self._bars:
            if start <= b.timestamp <= end:
                rows.append({
                    "open": b.open, "high": b.high, "low": b.low,
                    "close": b.close, "volume": b.volume,
                    "timestamp": b.timestamp,
                })
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).set_index("timestamp")


def _make_trending_bars(n: int = 50, symbol: str = "TEST") -> list[Bar]:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    bars = []
    price = 100.0
    for i, dt in enumerate(dates):
        if i < n // 2:
            price += 0.5
        else:
            price -= 0.3
        bars.append(Bar(
            timestamp=dt.to_pydatetime(),
            open=price - 0.1, high=price + 0.5, low=price - 0.5,
            close=price, volume=1000.0, symbol=symbol,
        ))
    return bars


class TrendStrategy(Strategy):
    """Simple strategy: buy after `warmup` bars, sell after `hold` more bars."""

    def __init__(self, broker, portfolio, warmup=5, hold=10):
        super().__init__(broker, portfolio)
        self.warmup = warmup
        self.hold = hold
        self._count = 0
        self._bought = False

    def on_bar(self, bar: Bar) -> None:
        self._count += 1
        if self._count == self.warmup and not self._bought:
            self.submit_order(bar.symbol, "buy", 10.0)
            self._bought = True
        elif self._count == self.warmup + self.hold and self._bought:
            self.submit_order(bar.symbol, "sell", 10.0)
            self._bought = False


# ---------- ParameterSpace ----------

class TestParameterSpace:
    def test_range_int(self):
        ps = ParameterSpace.range("x", 1, 5, 1)
        assert ps.values == [1, 2, 3, 4]

    def test_range_float(self):
        ps = ParameterSpace.range("x", 0.1, 0.4, 0.1)
        assert len(ps.values) == 3
        assert ps.values[0] == pytest.approx(0.1)

    def test_choices(self):
        ps = ParameterSpace.choices("mode", ["fast", "slow", "medium"])
        assert ps.values == ["fast", "slow", "medium"]


# ---------- Grid Search ----------

class TestGridSearch:
    def _make_factory(self, bars):
        inst = equity("TEST")

        def factory(params):
            feed = MockDataFeed(bars)
            broker = SimulatedBroker(commission_rate=0.0)
            portfolio = Portfolio(initial_cash=100_000.0)
            strategy = TrendStrategy(
                broker, portfolio,
                warmup=params["warmup"], hold=params["hold"],
            )
            return BacktestEngine(
                data_feed=feed, strategy=strategy, broker=broker,
                instruments=[inst],
                start=datetime(2024, 1, 1), end=datetime(2024, 2, 19),
            )
        return factory

    def test_grid_search_runs(self):
        bars = _make_trending_bars(50)
        factory = self._make_factory(bars)
        spaces = [
            ParameterSpace.choices("warmup", [3, 5]),
            ParameterSpace.choices("hold", [5, 10]),
        ]
        optimizer = Optimizer(factory, spaces, lambda m: m["sharpe_ratio"])
        results = optimizer.grid_search()
        assert len(results) == 4  # 2 * 2

    def test_best_returns_highest_objective(self):
        bars = _make_trending_bars(50)
        factory = self._make_factory(bars)
        spaces = [
            ParameterSpace.choices("warmup", [3, 5, 10]),
            ParameterSpace.choices("hold", [5, 10, 15]),
        ]
        optimizer = Optimizer(factory, spaces, lambda m: m["sharpe_ratio"])
        optimizer.grid_search()
        best = optimizer.best()
        assert best is not None
        assert best.objective_value == max(r.objective_value for r in optimizer.results)


# ---------- Random Search ----------

class TestRandomSearch:
    def test_random_search_n_trials(self):
        bars = _make_trending_bars(50)
        inst = equity("TEST")

        def factory(params):
            feed = MockDataFeed(bars)
            broker = SimulatedBroker(commission_rate=0.0)
            portfolio = Portfolio(initial_cash=100_000.0)
            strategy = TrendStrategy(broker, portfolio, warmup=params["warmup"], hold=params["hold"])
            return BacktestEngine(
                data_feed=feed, strategy=strategy, broker=broker,
                instruments=[inst],
                start=datetime(2024, 1, 1), end=datetime(2024, 2, 19),
            )

        spaces = [
            ParameterSpace.range("warmup", 3, 15, 1),
            ParameterSpace.range("hold", 5, 20, 1),
        ]
        optimizer = Optimizer(factory, spaces, lambda m: m["total_return"])
        results = optimizer.random_search(n_trials=5)
        assert len(results) == 5


# ---------- Walk-Forward ----------

class TestWalkForward:
    def test_window_creation(self):
        bars = _make_trending_bars(100, "TEST")
        inst = equity("TEST")

        def factory(params, start, end):
            feed = MockDataFeed(bars)
            broker = SimulatedBroker(commission_rate=0.0)
            portfolio = Portfolio(initial_cash=100_000.0)
            strategy = TrendStrategy(broker, portfolio, warmup=params["warmup"], hold=params["hold"])
            return BacktestEngine(
                data_feed=feed, strategy=strategy, broker=broker,
                instruments=[inst], start=start, end=end,
            )

        spaces = [
            ParameterSpace.choices("warmup", [3, 5]),
            ParameterSpace.choices("hold", [5, 10]),
        ]
        wf = WalkForwardOptimizer(
            engine_factory=factory,
            param_spaces=spaces,
            objective=lambda m: m["sharpe_ratio"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 4, 9),
            n_windows=3,
            train_ratio=0.7,
        )
        result = wf.run()

        assert len(result.windows) == 3
        for w in result.windows:
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert w.train_end == w.test_start

    def test_anchored_mode(self):
        bars = _make_trending_bars(100, "TEST")
        inst = equity("TEST")

        def factory(params, start, end):
            feed = MockDataFeed(bars)
            broker = SimulatedBroker(commission_rate=0.0)
            portfolio = Portfolio(initial_cash=100_000.0)
            strategy = TrendStrategy(broker, portfolio, warmup=params["warmup"], hold=params["hold"])
            return BacktestEngine(
                data_feed=feed, strategy=strategy, broker=broker,
                instruments=[inst], start=start, end=end,
            )

        spaces = [ParameterSpace.choices("warmup", [5]), ParameterSpace.choices("hold", [10])]
        wf = WalkForwardOptimizer(
            engine_factory=factory,
            param_spaces=spaces,
            objective=lambda m: m["total_return"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 4, 9),
            n_windows=2,
            train_ratio=0.7,
            anchored=True,
        )
        result = wf.run()

        # In anchored mode, all windows start training from the same start date
        for w in result.windows:
            assert w.train_start == datetime(2024, 1, 1)

    def test_aggregate_metrics(self):
        bars = _make_trending_bars(100, "TEST")
        inst = equity("TEST")

        def factory(params, start, end):
            feed = MockDataFeed(bars)
            broker = SimulatedBroker(commission_rate=0.0)
            portfolio = Portfolio(initial_cash=100_000.0)
            strategy = TrendStrategy(broker, portfolio, warmup=params["warmup"], hold=params["hold"])
            return BacktestEngine(
                data_feed=feed, strategy=strategy, broker=broker,
                instruments=[inst], start=start, end=end,
            )

        spaces = [ParameterSpace.choices("warmup", [5]), ParameterSpace.choices("hold", [10])]
        wf = WalkForwardOptimizer(
            engine_factory=factory,
            param_spaces=spaces,
            objective=lambda m: m["total_return"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 4, 9),
            n_windows=2,
            train_ratio=0.7,
        )
        result = wf.run()

        assert isinstance(result.avg_test_objective, float)
        assert len(result.test_objectives) == 2
        assert len(result.best_params_per_window) == 2
