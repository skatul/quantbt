from __future__ import annotations

from quantbt.indicators.base import Indicator


class VWAP(Indicator):
    """Volume-Weighted Average Price — cumulative intraday."""

    def __init__(self) -> None:
        self._cum_vol: float = 0.0
        self._cum_pv: float = 0.0
        self._value: float | None = None

    def update(self, value: float) -> float | None:
        """Close-only update — requires update_bar for proper VWAP."""
        return self.update_pv(value, 1.0)

    def update_pv(self, typical_price: float, volume: float) -> float | None:
        self._cum_vol += volume
        self._cum_pv += typical_price * volume
        if self._cum_vol > 0:
            self._value = self._cum_pv / self._cum_vol
        return self._value

    def update_bar(self, bar) -> float | None:
        typical = (bar.high + bar.low + bar.close) / 3.0
        return self.update_pv(typical, bar.volume)

    def reset(self) -> None:
        self._cum_vol = 0.0
        self._cum_pv = 0.0
        self._value = None

    @property
    def ready(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> float | None:
        return self._value


class OBV(Indicator):
    """On-Balance Volume."""

    def __init__(self) -> None:
        self._prev_close: float | None = None
        self._value: float = 0.0
        self._started: bool = False

    def update(self, value: float) -> float | None:
        """Close-only — requires update_cv for proper OBV."""
        return self.update_cv(value, 1.0)

    def update_cv(self, close: float, volume: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            self._value = volume
            self._started = True
            return self._value

        if close > self._prev_close:
            self._value += volume
        elif close < self._prev_close:
            self._value -= volume
        self._prev_close = close
        return self._value

    def update_bar(self, bar) -> float | None:
        return self.update_cv(bar.close, bar.volume)

    def reset(self) -> None:
        self._prev_close = None
        self._value = 0.0
        self._started = False

    @property
    def ready(self) -> bool:
        return self._started

    @property
    def value(self) -> float | None:
        return self._value if self._started else None
