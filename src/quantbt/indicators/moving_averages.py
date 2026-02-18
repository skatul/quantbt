from __future__ import annotations

from collections import deque

from quantbt.indicators.base import Indicator


class SMA(Indicator):
    """Simple Moving Average — streaming, deque-based."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._buffer: deque[float] = deque(maxlen=period)
        self._sum: float = 0.0
        self._value: float | None = None

    def update(self, value: float) -> float | None:
        if len(self._buffer) == self.period:
            self._sum -= self._buffer[0]
        self._buffer.append(value)
        self._sum += value
        if len(self._buffer) == self.period:
            self._value = self._sum / self.period
            return self._value
        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._sum = 0.0
        self._value = None

    @property
    def ready(self) -> bool:
        return len(self._buffer) == self.period

    @property
    def value(self) -> float | None:
        return self._value


class EMA(Indicator):
    """Exponential Moving Average — Wilder smoothing."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._k: float = 2.0 / (period + 1)
        self._count: int = 0
        self._sum: float = 0.0
        self._value: float | None = None

    def update(self, value: float) -> float | None:
        self._count += 1
        if self._count < self.period:
            self._sum += value
            return None
        if self._count == self.period:
            self._sum += value
            self._value = self._sum / self.period
            return self._value
        # Subsequent values: EMA = prev + k * (value - prev)
        self._value = self._value + self._k * (value - self._value)  # type: ignore[operator]
        return self._value

    def reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._value = None

    @property
    def ready(self) -> bool:
        return self._count >= self.period

    @property
    def value(self) -> float | None:
        return self._value
