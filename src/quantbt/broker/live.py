from __future__ import annotations

import threading

from quantbt.broker.base import Broker
from quantbt.instrument.model import Instrument
from quantbt.messaging.protocol import Message, message_from_dict, new_order_message
from quantbt.messaging.zmq_client import ZmqClient
from quantbt.orders.model import Fill, Order, OrderStatus


class LiveBroker(Broker):
    """Broker that sends orders to tradecore via ZMQ."""

    def __init__(self, zmq_address: str = "tcp://127.0.0.1:5555") -> None:
        super().__init__()
        self._client = ZmqClient(zmq_address)
        self._pending_orders: dict[str, Order] = {}  # cl_ord_id -> Order

    def submit_order(self, order: Order) -> None:
        # Build instrument dict from symbol (minimal for MVP)
        instrument_dict = {"symbol": order.symbol, "asset_class": "equity"}

        msg = new_order_message(
            cl_ord_id=order.cl_ord_id,
            instrument_dict=instrument_dict,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            time_in_force=order.time_in_force.value,
            strategy_id=order.strategy_id,
        )
        self._client.send(msg)
        self._pending_orders[order.cl_ord_id] = order

    def poll_responses(self, timeout_ms: int = 100) -> None:
        raw = self._client.recv(timeout_ms=timeout_ms)
        if raw is None:
            return
        msg = message_from_dict(raw)
        self._dispatch(msg)

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._pending_orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.ACCEPTED)]

    def cancel_order(self, cl_ord_id: str) -> bool:
        # TODO: send cancel message to tradecore
        return False

    def close(self) -> None:
        self._client.close()

    def _dispatch(self, msg: Message) -> None:
        cl_ord_id = msg.payload.get("cl_ord_id", "")
        order = self._pending_orders.get(cl_ord_id)
        if order is None:
            return

        if msg.msg_type == "order_ack":
            order.order_id = msg.payload.get("order_id")
            order.status = OrderStatus.ACCEPTED

        elif msg.msg_type == "fill":
            order.status = OrderStatus(msg.payload.get("status", "filled"))
            fill = Fill(
                fill_id=msg.payload["fill_id"],
                order_id=msg.payload["order_id"],
                cl_ord_id=cl_ord_id,
                fill_price=msg.payload["fill_price"],
                fill_quantity=msg.payload["fill_quantity"],
                remaining_quantity=msg.payload["remaining_quantity"],
                commission=msg.payload.get("commission", 0.0),
            )
            self._notify_fill(order, fill)
            if order.status == OrderStatus.FILLED:
                del self._pending_orders[cl_ord_id]

        elif msg.msg_type == "reject":
            order.status = OrderStatus.REJECTED
            del self._pending_orders[cl_ord_id]
