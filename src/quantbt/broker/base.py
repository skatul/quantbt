from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from quantbt.orders.model import Fill, Order


class Broker(ABC):
    def __init__(self) -> None:
        self._fill_callbacks: list[Callable[[Order, Fill], None]] = []

    def on_fill(self, callback: Callable[[Order, Fill], None]) -> None:
        self._fill_callbacks.append(callback)

    def _notify_fill(self, order: Order, fill: Fill) -> None:
        for cb in self._fill_callbacks:
            cb(order, fill)

    @abstractmethod
    def submit_order(self, order: Order) -> None: ...

    @abstractmethod
    def get_open_orders(self) -> list[Order]: ...

    @abstractmethod
    def cancel_order(self, cl_ord_id: str) -> bool: ...
