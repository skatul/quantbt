from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

import pandas as pd

from quantbt.data.bar import Bar
from quantbt.instrument.model import Instrument


class DataFeed(ABC):
    @abstractmethod
    def fetch(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Returns DataFrame with columns: open, high, low, close, volume, indexed by datetime."""
        ...

    def iter_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        df = self.fetch(instrument, start, end)
        for ts, row in df.iterrows():
            yield Bar(
                timestamp=ts.to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                symbol=instrument.symbol,
            )

    def iter_bars_sync(
        self,
        instruments: list[Instrument],
        start: datetime,
        end: datetime,
    ) -> Iterator[tuple[datetime, dict[str, Bar]]]:
        """Yield time-aligned bars across instruments, merged by timestamp.

        Returns ``(timestamp, {symbol: Bar})`` tuples in chronological order.
        At each timestamp, only instruments that have a bar at that time are included.
        """
        iterators: dict[str, Iterator[Bar]] = {}
        for inst in instruments:
            iterators[inst.symbol] = self.iter_bars(inst, start, end)

        # Seed the heap with the first bar from each iterator
        heap: list[tuple[datetime, str, Bar]] = []
        for symbol, it in iterators.items():
            bar = next(it, None)
            if bar is not None:
                heap.append((bar.timestamp, symbol, bar))
        heapq.heapify(heap)

        # Merge bars sharing the same timestamp
        while heap:
            ts = heap[0][0]
            bars: dict[str, Bar] = {}
            while heap and heap[0][0] == ts:
                _, symbol, bar = heapq.heappop(heap)
                bars[symbol] = bar
                nxt = next(iterators[symbol], None)
                if nxt is not None:
                    heapq.heappush(heap, (nxt.timestamp, symbol, nxt))
            yield ts, bars
