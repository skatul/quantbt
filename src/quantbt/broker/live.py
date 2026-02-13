from __future__ import annotations

from quantbt.broker.base import Broker
from quantbt.data.bar import Bar
from quantbt.messaging import fix_messages_pb2 as fix
from quantbt.messaging.protocol import new_order_message
from quantbt.messaging.zmq_client import ZmqClient
from quantbt.orders.model import Fill, Order, OrderStatus


class LiveBroker(Broker):
    """Broker that sends orders to tradecore via ZMQ."""

    def __init__(self, zmq_address: str = "tcp://127.0.0.1:5555") -> None:
        super().__init__()
        self._client = ZmqClient(zmq_address)
        self._pending_orders: dict[str, Order] = {}  # cl_ord_id -> Order
        self._current_bar: Bar | None = None

    def set_current_bar(self, bar: Bar) -> None:
        """Set the current bar so market orders include a price hint."""
        self._current_bar = bar

    def submit_order(self, order: Order) -> None:
        # Build instrument dict from symbol (minimal for MVP)
        instrument_dict = {"symbol": order.symbol, "asset_class": "equity"}

        # Include current market price so tradecore can fill market orders
        market_price = None
        if self._current_bar is not None and self._current_bar.symbol == order.symbol:
            market_price = self._current_bar.close

        msg = new_order_message(
            cl_ord_id=order.cl_ord_id,
            instrument_dict=instrument_dict,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            time_in_force=order.time_in_force.value,
            strategy_id=order.strategy_id,
            market_price=market_price,
        )
        self._client.send(msg)
        self._pending_orders[order.cl_ord_id] = order

    def poll_responses(self, timeout_ms: int = 100) -> None:
        response = self._client.recv(timeout_ms=timeout_ms)
        if response is None:
            return
        self._dispatch(response)

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._pending_orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.ACCEPTED)]

    def cancel_order(self, cl_ord_id: str) -> bool:
        # TODO: send cancel message to tradecore
        return False

    def close(self) -> None:
        self._client.close()

    def _dispatch(self, msg: fix.FixMessage) -> None:
        if msg.HasField("execution_report"):
            er = msg.execution_report
            cl_ord_id = er.cl_ord_id
            order = self._pending_orders.get(cl_ord_id)
            if order is None:
                return

            if er.exec_type == fix.EXEC_TYPE_NEW:
                order.order_id = er.order_id
                order.status = OrderStatus.ACCEPTED

            elif er.exec_type in (fix.EXEC_TYPE_FILL, fix.EXEC_TYPE_PARTIAL_FILL):
                if er.ord_status == fix.ORD_STATUS_FILLED:
                    order.status = OrderStatus.FILLED
                elif er.ord_status == fix.ORD_STATUS_PARTIALLY_FILLED:
                    order.status = OrderStatus.PARTIALLY_FILLED

                order.order_id = er.order_id
                fill = Fill(
                    fill_id=er.exec_id,
                    order_id=er.order_id,
                    cl_ord_id=cl_ord_id,
                    fill_price=er.last_px,
                    fill_quantity=er.last_qty,
                    remaining_quantity=er.leaves_qty,
                    commission=er.commission,
                )
                self._notify_fill(order, fill)
                if order.status == OrderStatus.FILLED:
                    del self._pending_orders[cl_ord_id]

        elif msg.HasField("reject"):
            # Try to find order by ref_msg_seq_num - for now, reject all pending
            # In a production system we'd correlate by msg_seq_num
            rej_text = msg.reject.text
            # Find any pending order (best-effort matching)
            for cl_ord_id, order in list(self._pending_orders.items()):
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.REJECTED
                    del self._pending_orders[cl_ord_id]
                    break
