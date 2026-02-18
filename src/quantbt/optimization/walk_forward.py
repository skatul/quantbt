from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from quantbt.core.engine import BacktestEngine
from quantbt.optimization.optimizer import Optimizer, OptimizationResult
from quantbt.optimization.parameter import ParameterSpace


@dataclass
class WindowResult:
    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_objective: float
    test_objective: float


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]

    @property
    def avg_test_objective(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.test_objective for w in self.windows) / len(self.windows)

    @property
    def test_objectives(self) -> list[float]:
        return [w.test_objective for w in self.windows]

    @property
    def best_params_per_window(self) -> list[dict]:
        return [w.best_params for w in self.windows]


class WalkForwardOptimizer:
    """Walk-forward optimization with train/test window splitting.

    Splits the total time range into ``n_windows`` sequential windows.
    For each window, optimizes parameters on the training portion and
    validates on the test portion.
    """

    def __init__(
        self,
        engine_factory: Callable[[dict, datetime, datetime], BacktestEngine],
        param_spaces: list[ParameterSpace],
        objective: Callable[[dict[str, float]], float],
        start: datetime,
        end: datetime,
        n_windows: int = 5,
        train_ratio: float = 0.7,
        anchored: bool = False,
    ) -> None:
        self.engine_factory = engine_factory
        self.param_spaces = param_spaces
        self.objective = objective
        self.start = start
        self.end = end
        self.n_windows = n_windows
        self.train_ratio = train_ratio
        self.anchored = anchored

    def _split_windows(self) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        total = self.end - self.start
        window_size = total / self.n_windows
        windows = []
        for i in range(self.n_windows):
            w_start = self.start + window_size * i
            w_end = self.start + window_size * (i + 1)
            train_start = self.start if self.anchored else w_start
            split = w_start + (w_end - w_start) * self.train_ratio
            train_end = split
            test_start = split
            test_end = w_end
            windows.append((train_start, train_end, test_start, test_end))
        return windows

    def run(self) -> WalkForwardResult:
        windows = self._split_windows()
        results: list[WindowResult] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Optimize on train period
            train_factory = lambda params, ts=train_start, te=train_end: \
                self.engine_factory(params, ts, te)
            optimizer = Optimizer(train_factory, self.param_spaces, self.objective)
            optimizer.grid_search()
            best = optimizer.best()
            if best is None:
                continue

            # Validate on test period
            test_engine = self.engine_factory(best.params, test_start, test_end)
            test_engine.run()
            test_metrics = test_engine.performance.summary()
            test_obj = self.objective(test_metrics)

            results.append(WindowResult(
                window_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best.params,
                train_metrics=best.metrics,
                test_metrics=test_metrics,
                train_objective=best.objective_value,
                test_objective=test_obj,
            ))

        return WalkForwardResult(windows=results)
