from __future__ import annotations

from uuid import uuid4

from quantbt.broker.base import Broker
from quantbt.data.bar import Bar
from quantbt.orders.model import Fill, Order, OrderStatus, OrderType


class SimulatedBroker(Broker):
    """Fills market orders immediately at the current bar's close price."""

    def __init__(self, commission_rate: float = 0.001, slippage_bps: float = 0.0) -> None:
        super().__init__()
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self._pending_orders: list[Order] = []
        self._current_bar: Bar | None = None
        self._current_bars: dict[str, Bar] = {}
        self._order_seq = 0

    def set_current_bar(self, bar: Bar) -> None:
        self._current_bar = bar
        self._current_bars[bar.symbol] = bar
        self._process_pending_orders()

    def submit_order(self, order: Order) -> None:
        self._order_seq += 1
        order.order_id = f"SIM-{self._order_seq:05d}"
        order.status = OrderStatus.ACCEPTED

        if order.order_type == OrderType.MARKET and self._current_bar is not None:
            self._fill_order(order, self._current_bar.close)
        else:
            self._pending_orders.append(order)

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._pending_orders if o.status == OrderStatus.ACCEPTED]

    def cancel_order(self, cl_ord_id: str) -> bool:
        for order in self._pending_orders:
            if order.cl_ord_id == cl_ord_id and order.status == OrderStatus.ACCEPTED:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def _process_pending_orders(self) -> None:
        if self._current_bar is None:
            return
        remaining: list[Order] = []
        for order in self._pending_orders:
            if order.status != OrderStatus.ACCEPTED:
                continue
            if order.order_type == OrderType.MARKET:
                self._fill_order(order, self._current_bar.close)
            elif order.order_type == OrderType.LIMIT:
                if (order.side.value == "buy" and self._current_bar.low <= order.limit_price) or \
                   (order.side.value == "sell" and self._current_bar.high >= order.limit_price):
                    self._fill_order(order, order.limit_price)
                else:
                    remaining.append(order)
            else:
                remaining.append(order)
        self._pending_orders = remaining

    def _fill_order(self, order: Order, price: float) -> None:
        if self.slippage_bps != 0.0:
            if order.side.value == "buy":
                price = price * (1.0 + self.slippage_bps / 10000.0)
            else:
                price = price * (1.0 - self.slippage_bps / 10000.0)
        commission = price * order.quantity * self.commission_rate
        fill = Fill(
            fill_id=str(uuid4())[:8],
            order_id=order.order_id or "",
            cl_ord_id=order.cl_ord_id,
            fill_price=price,
            fill_quantity=order.quantity,
            remaining_quantity=0.0,
            commission=commission,
        )
        order.status = OrderStatus.FILLED
        self._notify_fill(order, fill)
