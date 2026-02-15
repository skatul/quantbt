from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from quantbt.messaging import fix_messages_pb2 as fix


def generate_uuid() -> str:
    return str(uuid4())


def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


# ---- Enum mappings ----

_SIDE_MAP = {"buy": fix.SIDE_BUY, "sell": fix.SIDE_SELL}
_ORD_TYPE_MAP = {"market": fix.ORD_TYPE_MARKET, "limit": fix.ORD_TYPE_LIMIT}
_TIF_MAP = {"day": fix.TIF_DAY, "gtc": fix.TIF_GTC, "ioc": fix.TIF_IOC}
_ASSET_CLASS_MAP = {
    "equity": fix.SECURITY_TYPE_COMMON_STOCK,
    "future": fix.SECURITY_TYPE_FUTURE,
    "option": fix.SECURITY_TYPE_OPTION,
    "fx": fix.SECURITY_TYPE_FX_SPOT,
}


# ---- Builders ----

def new_order_message(
    cl_ord_id: str,
    instrument_dict: dict,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "day",
    strategy_id: str = "",
    market_price: float | None = None,
) -> fix.FixMessage:
    msg = fix.FixMessage()
    msg.sender_comp_id = "QUANTBT"
    msg.target_comp_id = "TRADECORE"
    msg.msg_seq_num = generate_uuid()
    msg.sending_time = current_timestamp()

    nos = msg.new_order_single
    nos.cl_ord_id = cl_ord_id
    nos.instrument.symbol = instrument_dict.get("symbol", "")
    nos.instrument.security_type = _ASSET_CLASS_MAP.get(
        instrument_dict.get("asset_class", "equity"), fix.SECURITY_TYPE_COMMON_STOCK
    )
    if instrument_dict.get("exchange"):
        nos.instrument.exchange = instrument_dict["exchange"]
    if instrument_dict.get("currency"):
        nos.instrument.currency = instrument_dict["currency"]
    nos.side = _SIDE_MAP.get(side, fix.SIDE_BUY)
    nos.order_qty = quantity
    nos.ord_type = _ORD_TYPE_MAP.get(order_type, fix.ORD_TYPE_MARKET)
    if limit_price is not None:
        nos.price = limit_price
    nos.time_in_force = _TIF_MAP.get(time_in_force, fix.TIF_DAY)
    nos.text = strategy_id
    nos.transact_time = current_timestamp()
    if market_price is not None:
        nos.market_price = market_price

    return msg


def cancel_order_message(
    orig_cl_ord_id: str,
    instrument_dict: dict,
    side: str,
    quantity: float,
) -> fix.FixMessage:
    msg = fix.FixMessage()
    msg.sender_comp_id = "QUANTBT"
    msg.target_comp_id = "TRADECORE"
    msg.msg_seq_num = generate_uuid()
    msg.sending_time = current_timestamp()

    cancel = msg.order_cancel_request
    cancel.cl_ord_id = generate_uuid()
    cancel.orig_cl_ord_id = orig_cl_ord_id
    cancel.instrument.symbol = instrument_dict.get("symbol", "")
    cancel.instrument.security_type = _ASSET_CLASS_MAP.get(
        instrument_dict.get("asset_class", "equity"), fix.SECURITY_TYPE_COMMON_STOCK
    )
    cancel.side = _SIDE_MAP.get(side, fix.SIDE_BUY)
    cancel.order_qty = quantity
    cancel.transact_time = current_timestamp()

    return msg


def position_query_message(strategy_id: str = "") -> fix.FixMessage:
    msg = fix.FixMessage()
    msg.sender_comp_id = "QUANTBT"
    msg.target_comp_id = "TRADECORE"
    msg.msg_seq_num = generate_uuid()
    msg.sending_time = current_timestamp()

    pr = msg.position_request
    pr.pos_req_id = generate_uuid()
    if strategy_id:
        pr.account = strategy_id

    return msg


def heartbeat_message() -> fix.FixMessage:
    msg = fix.FixMessage()
    msg.sender_comp_id = "QUANTBT"
    msg.target_comp_id = "TRADECORE"
    msg.msg_seq_num = generate_uuid()
    msg.sending_time = current_timestamp()

    msg.heartbeat.test_req_id = generate_uuid()

    return msg


# ---- Serialization ----

def serialize(msg: fix.FixMessage) -> bytes:
    return msg.SerializeToString()


def deserialize(data: bytes) -> fix.FixMessage:
    msg = fix.FixMessage()
    msg.ParseFromString(data)
    return msg
