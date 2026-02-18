from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from quantbt.core.engine import BacktestEngine
from quantbt.optimization.parameter import ParameterSpace


@dataclass
class OptimizationResult:
    params: dict
    objective_value: float
    metrics: dict[str, float]


class Optimizer:
    """Runs parameter optimization over a BacktestEngine.

    ``engine_factory`` takes a parameter dict and returns a fully configured
    BacktestEngine ready to run.

    ``objective`` takes a dict of performance metrics and returns a scalar to
    maximize (e.g., Sharpe ratio).
    """

    def __init__(
        self,
        engine_factory: Callable[[dict], BacktestEngine],
        param_spaces: list[ParameterSpace],
        objective: Callable[[dict[str, float]], float],
    ) -> None:
        self.engine_factory = engine_factory
        self.param_spaces = param_spaces
        self.objective = objective
        self.results: list[OptimizationResult] = []

    def _evaluate(self, params: dict) -> OptimizationResult:
        engine = self.engine_factory(params)
        engine.run()
        metrics = engine.performance.summary()
        obj = self.objective(metrics)
        result = OptimizationResult(params=params, objective_value=obj, metrics=metrics)
        self.results.append(result)
        return result

    def grid_search(self) -> list[OptimizationResult]:
        """Exhaustive search over all parameter combinations."""
        names = [ps.name for ps in self.param_spaces]
        value_lists = [ps.values for ps in self.param_spaces]
        results: list[OptimizationResult] = []
        for combo in itertools.product(*value_lists):
            params = dict(zip(names, combo))
            results.append(self._evaluate(params))
        return results

    def random_search(self, n_trials: int) -> list[OptimizationResult]:
        """Random sampling of parameter combinations."""
        names = [ps.name for ps in self.param_spaces]
        value_lists = [ps.values for ps in self.param_spaces]
        results: list[OptimizationResult] = []
        for _ in range(n_trials):
            combo = tuple(random.choice(vals) for vals in value_lists)
            params = dict(zip(names, combo))
            results.append(self._evaluate(params))
        return results

    def best(self) -> OptimizationResult | None:
        """Return the result with the highest objective value."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.objective_value)
