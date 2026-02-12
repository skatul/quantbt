from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Message:
    msg_type: str
    payload: dict
    msg_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ref_msg_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        return cls(**d)


def new_order_message(
    cl_ord_id: str,
    instrument_dict: dict,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "day",
    strategy_id: str = "",
) -> Message:
    return Message(
        msg_type="new_order",
        payload={
            "cl_ord_id": cl_ord_id,
            "instrument": instrument_dict,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "time_in_force": time_in_force,
            "strategy_id": strategy_id,
        },
    )


def position_query_message(strategy_id: str = "") -> Message:
    return Message(
        msg_type="position_query",
        payload={"strategy_id": strategy_id},
    )


def heartbeat_message() -> Message:
    return Message(msg_type="heartbeat", payload={})


MSG_TYPE_REGISTRY = {
    "new_order",
    "order_ack",
    "fill",
    "reject",
    "position_query",
    "position_report",
    "heartbeat",
}


def message_from_dict(d: dict) -> Message:
    return Message.from_dict(d)
