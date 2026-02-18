from __future__ import annotations

import math
from collections import deque

from quantbt.indicators.base import Indicator
from quantbt.indicators.moving_averages import SMA


class BollingerBands(Indicator):
    """Bollinger Bands — middle (SMA), upper, lower bands."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self.period = period
        self.num_std = num_std
        self._buffer: deque[float] = deque(maxlen=period)
        self._sma = SMA(period)
        self._middle: float | None = None
        self._upper: float | None = None
        self._lower: float | None = None

    def update(self, value: float) -> float | None:
        self._buffer.append(value)
        mid = self._sma.update(value)
        if mid is None:
            return None

        self._middle = mid
        variance = sum((v - mid) ** 2 for v in self._buffer) / self.period
        std = math.sqrt(variance)
        self._upper = mid + self.num_std * std
        self._lower = mid - self.num_std * std
        return self._middle

    def reset(self) -> None:
        self._buffer.clear()
        self._sma.reset()
        self._middle = None
        self._upper = None
        self._lower = None

    @property
    def ready(self) -> bool:
        return self._middle is not None

    @property
    def value(self) -> float | None:
        return self._middle

    @property
    def middle(self) -> float | None:
        return self._middle

    @property
    def upper(self) -> float | None:
        return self._upper

    @property
    def lower(self) -> float | None:
        return self._lower

    @property
    def upper(self) -> float | None:
        return self._upper

    @property
    def lower(self) -> float | None:
        return self._lower


class ATR(Indicator):
    """Average True Range."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._count: int = 0
        self._prev_close: float | None = None
        self._tr_values: list[float] = []
        self._value: float | None = None

    def update(self, value: float) -> float | None:
        """For close-only streaming — true range = |close - prev_close|."""
        if self._prev_close is None:
            self._prev_close = value
            return None
        tr = abs(value - self._prev_close)
        self._prev_close = value
        return self._update_tr(tr)

    def update_bar(self, bar) -> float | None:
        """Feed a Bar with high, low, close for proper ATR calculation."""
        if self._prev_close is None:
            self._prev_close = bar.close
            return None
        tr = max(
            bar.high - bar.low,
            abs(bar.high - self._prev_close),
            abs(bar.low - self._prev_close),
        )
        self._prev_close = bar.close
        return self._update_tr(tr)

    def _update_tr(self, tr: float) -> float | None:
        self._count += 1
        if self._count <= self.period:
            self._tr_values.append(tr)
            if self._count == self.period:
                self._value = sum(self._tr_values) / self.period
                return self._value
            return None
        # Wilder smoothing
        self._value = (self._value * (self.period - 1) + tr) / self.period  # type: ignore[operator]
        return self._value

    def reset(self) -> None:
        self._count = 0
        self._prev_close = None
        self._tr_values = []
        self._value = None

    @property
    def ready(self) -> bool:
        return self._count >= self.period

    @property
    def value(self) -> float | None:
        return self._value
