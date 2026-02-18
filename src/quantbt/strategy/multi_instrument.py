from __future__ import annotations

from abc import abstractmethod

from quantbt.data.bar import Bar
from quantbt.strategy.base import Strategy


class MultiInstrumentStrategy(Strategy):
    """Strategy that operates on multiple instruments simultaneously.

    Override ``on_bars`` to receive all bars for a given timestamp.
    ``on_bar`` is a no-op — the engine will call ``on_bars`` instead.
    """

    def on_bar(self, bar: Bar) -> None:
        # No-op: multi-instrument strategies use on_bars
        pass

    @abstractmethod
    def on_bars(self, bars: dict[str, Bar]) -> None:
        """Called with {symbol: Bar} for each synchronized timestamp."""
        ...
