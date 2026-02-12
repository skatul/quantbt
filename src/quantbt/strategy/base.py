from __future__ import annotations

from abc import ABC, abstractmethod

from quantbt.broker.base import Broker
from quantbt.data.bar import Bar
from quantbt.orders.model import Fill, Order, OrderType, Side
from quantbt.portfolio.portfolio import Portfolio


class Strategy(ABC):
    """Event-driven strategy base. Subclass and implement on_bar."""

    def __init__(self, broker: Broker, portfolio: Portfolio) -> None:
        self.broker = broker
        self.portfolio = portfolio

    def on_init(self) -> None:
        """Called once before backtest starts. Override for indicator setup."""

    @abstractmethod
    def on_bar(self, bar: Bar) -> None:
        """Called for each new bar. Generate signals and submit orders here."""
        ...

    def on_fill(self, order: Order, fill: Fill) -> None:
        """Called when an order is filled. Override for custom logic."""

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> str:
        """Submit an order via the broker. Returns cl_ord_id."""
        cl_ord_id = Order.generate_cl_ord_id()
        order = Order(
            cl_ord_id=cl_ord_id,
            symbol=symbol,
            side=Side(side),
            quantity=quantity,
            order_type=OrderType(order_type),
            limit_price=limit_price,
            strategy_id=self.__class__.__name__,
        )
        self.broker.submit_order(order)
        return cl_ord_id
