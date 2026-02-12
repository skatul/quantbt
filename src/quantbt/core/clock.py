from __future__ import annotations

from datetime import datetime, timezone


class SimulatedClock:
    """Tracks simulated time during backtesting."""

    def __init__(self) -> None:
        self._current_time: datetime | None = None

    @property
    def now(self) -> datetime:
        if self._current_time is None:
            return datetime.now(timezone.utc)
        return self._current_time

    def advance(self, timestamp: datetime) -> None:
        self._current_time = timestamp

    def reset(self) -> None:
        self._current_time = None
