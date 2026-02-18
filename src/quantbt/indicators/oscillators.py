from __future__ import annotations

from collections import deque

from quantbt.indicators.base import Indicator
from quantbt.indicators.moving_averages import EMA


class RSI(Indicator):
    """Relative Strength Index (Wilder smoothing)."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._count: int = 0
        self._prev: float | None = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._gains: list[float] = []
        self._losses: list[float] = []
        self._value: float | None = None

    def update(self, value: float) -> float | None:
        if self._prev is None:
            self._prev = value
            return None

        change = value - self._prev
        self._prev = value
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._count += 1

        if self._count <= self.period:
            self._gains.append(gain)
            self._losses.append(loss)
            if self._count == self.period:
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
                if self._avg_loss == 0:
                    self._value = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    self._value = 100.0 - 100.0 / (1.0 + rs)
                return self._value
            return None

        # Wilder smoothing
        self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
        self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0:
            self._value = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._value = 100.0 - 100.0 / (1.0 + rs)
        return self._value

    def reset(self) -> None:
        self._count = 0
        self._prev = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._gains = []
        self._losses = []
        self._value = None

    @property
    def ready(self) -> bool:
        return self._count >= self.period

    @property
    def value(self) -> float | None:
        return self._value


class MACD(Indicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self._fast_ema = EMA(fast)
        self._slow_ema = EMA(slow)
        self._signal_ema = EMA(signal)
        self._macd_line: float | None = None
        self._signal_line: float | None = None
        self._histogram: float | None = None

    def update(self, value: float) -> float | None:
        fast_val = self._fast_ema.update(value)
        slow_val = self._slow_ema.update(value)

        if fast_val is None or slow_val is None:
            return None

        self._macd_line = fast_val - slow_val
        sig = self._signal_ema.update(self._macd_line)

        if sig is not None:
            self._signal_line = sig
            self._histogram = self._macd_line - self._signal_line
        return self._macd_line

    def reset(self) -> None:
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema.reset()
        self._macd_line = None
        self._signal_line = None
        self._histogram = None

    @property
    def ready(self) -> bool:
        return self._macd_line is not None

    @property
    def value(self) -> float | None:
        return self._macd_line

    @property
    def signal_line(self) -> float | None:
        return self._signal_line

    @property
    def histogram(self) -> float | None:
        return self._histogram


class Stochastic(Indicator):
    """Stochastic Oscillator (%K and %D)."""

    def __init__(self, k_period: int = 14, d_period: int = 3) -> None:
        self.k_period = k_period
        self.d_period = d_period
        self._highs: deque[float] = deque(maxlen=k_period)
        self._lows: deque[float] = deque(maxlen=k_period)
        self._closes: deque[float] = deque(maxlen=k_period)
        self._k_values: deque[float] = deque(maxlen=d_period)
        self._k: float | None = None
        self._d: float | None = None

    def update(self, value: float) -> float | None:
        """For streaming with close-only data, uses value as high/low/close."""
        return self.update_hlc(value, value, value)

    def update_hlc(self, high: float, low: float, close: float) -> float | None:
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if len(self._highs) < self.k_period:
            return None

        highest = max(self._highs)
        lowest = min(self._lows)
        if highest == lowest:
            self._k = 50.0
        else:
            self._k = 100.0 * (close - lowest) / (highest - lowest)

        self._k_values.append(self._k)
        if len(self._k_values) == self.d_period:
            self._d = sum(self._k_values) / self.d_period

        return self._k

    def reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._k_values.clear()
        self._k = None
        self._d = None

    @property
    def ready(self) -> bool:
        return self._k is not None

    @property
    def value(self) -> float | None:
        return self._k

    @property
    def k(self) -> float | None:
        return self._k

    @property
    def d(self) -> float | None:
        return self._d
