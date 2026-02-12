from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    cl_ord_id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    order_id: str | None = None  # assigned by tradecore
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_cl_ord_id() -> str:
        return str(uuid4())[:8]


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    cl_ord_id: str
    fill_price: float
    fill_quantity: float
    remaining_quantity: float
    commission: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
