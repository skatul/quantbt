from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    drawdown: float


class PerformanceTracker:
    """Records equity curve and computes performance metrics."""

    def __init__(self, initial_cash: float = 100_000.0,
                 periods_per_year: float = 252.0) -> None:
        self.initial_cash = initial_cash
        self.periods_per_year = periods_per_year
        self._equity_curve: list[EquityPoint] = []
        self._peak_equity: float = initial_cash
        self._returns: list[float] = []
        self._prev_equity: float = initial_cash

    def record(self, timestamp: datetime, equity: float,
               cash: float, positions_value: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity

        point = EquityPoint(
            timestamp=timestamp,
            equity=equity,
            cash=cash,
            positions_value=positions_value,
            drawdown=drawdown,
        )
        self._equity_curve.append(point)

        # Record return
        if self._prev_equity > 0:
            ret = (equity - self._prev_equity) / self._prev_equity
            self._returns.append(ret)
        self._prev_equity = equity

    @property
    def equity_curve(self) -> list[EquityPoint]:
        return self._equity_curve

    def total_return(self) -> float:
        if not self._equity_curve:
            return 0.0
        final = self._equity_curve[-1].equity
        return (final - self.initial_cash) / self.initial_cash

    def max_drawdown(self) -> float:
        if not self._equity_curve:
            return 0.0
        return max(p.drawdown for p in self._equity_curve)

    def sharpe_ratio(self, risk_free_rate: float = 0.0,
                     periods_per_year: float | None = None) -> float:
        if periods_per_year is None:
            periods_per_year = self.periods_per_year
        if len(self._returns) < 2:
            return 0.0

        mean_ret = sum(self._returns) / len(self._returns)
        excess = mean_ret - risk_free_rate / periods_per_year

        variance = sum((r - mean_ret) ** 2 for r in self._returns) / (len(self._returns) - 1)
        std = math.sqrt(variance)

        if std == 0:
            return 0.0

        return (excess / std) * math.sqrt(periods_per_year)

    def sortino_ratio(self, risk_free_rate: float = 0.0,
                      periods_per_year: float | None = None) -> float:
        if periods_per_year is None:
            periods_per_year = self.periods_per_year
        if len(self._returns) < 2:
            return 0.0

        mean_ret = sum(self._returns) / len(self._returns)
        excess = mean_ret - risk_free_rate / periods_per_year

        downside = [r for r in self._returns if r < 0]
        if not downside:
            return float("inf") if excess > 0 else 0.0

        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var)

        if downside_std == 0:
            return 0.0

        return (excess / downside_std) * math.sqrt(periods_per_year)

    def win_rate(self) -> float:
        if not self._returns:
            return 0.0
        wins = sum(1 for r in self._returns if r > 0)
        return wins / len(self._returns)

    def summary(self) -> dict[str, float]:
        return {
            "total_return": self.total_return(),
            "max_drawdown": self.max_drawdown(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "win_rate": self.win_rate(),
            "num_bars": len(self._equity_curve),
        }
