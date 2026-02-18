from __future__ import annotations

from abc import ABC, abstractmethod

from quantbt.data.bar import Bar


class Indicator(ABC):
    """Base class for all streaming indicators."""

    @abstractmethod
    def update(self, value: float) -> float | None:
        """Feed a new value and return the indicator result, or None if not ready."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state."""
        ...

    @property
    @abstractmethod
    def ready(self) -> bool:
        """True when enough data has been fed to produce a valid value."""
        ...

    @property
    @abstractmethod
    def value(self) -> float | None:
        """Current indicator value, or None if not ready."""
        ...

    def update_bar(self, bar: Bar) -> float | None:
        """Convenience: feed bar.close and return result."""
        return self.update(bar.close)
