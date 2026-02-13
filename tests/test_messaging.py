from quantbt.messaging import fix_messages_pb2 as fix
from quantbt.messaging.protocol import (
    deserialize,
    heartbeat_message,
    new_order_message,
    position_query_message,
    serialize,
)


def test_serialize_deserialize_roundtrip():
    msg = new_order_message(
        cl_ord_id="test-001",
        instrument_dict={"symbol": "AAPL", "asset_class": "equity"},
        side="buy",
        quantity=100.0,
    )
    data = serialize(msg)
    restored = deserialize(data)

    assert restored.HasField("new_order_single")
    nos = restored.new_order_single
    assert nos.cl_ord_id == "test-001"
    assert nos.instrument.symbol == "AAPL"
    assert nos.side == fix.SIDE_BUY
    assert nos.order_qty == 100.0


def test_new_order_message():
    msg = new_order_message(
        cl_ord_id="test-001",
        instrument_dict={"symbol": "AAPL", "asset_class": "equity"},
        side="buy",
        quantity=100.0,
        market_price=150.0,
    )
    assert msg.HasField("new_order_single")
    nos = msg.new_order_single
    assert nos.cl_ord_id == "test-001"
    assert nos.instrument.symbol == "AAPL"
    assert nos.instrument.security_type == fix.SECURITY_TYPE_COMMON_STOCK
    assert nos.side == fix.SIDE_BUY
    assert nos.order_qty == 100.0
    assert nos.ord_type == fix.ORD_TYPE_MARKET
    assert nos.market_price == 150.0
    assert msg.sender_comp_id == "QUANTBT"
    assert msg.target_comp_id == "TRADECORE"


def test_new_order_limit():
    msg = new_order_message(
        cl_ord_id="test-002",
        instrument_dict={"symbol": "MSFT", "asset_class": "equity"},
        side="sell",
        quantity=50.0,
        order_type="limit",
        limit_price=300.0,
        time_in_force="gtc",
        strategy_id="sma_v1",
    )
    nos = msg.new_order_single
    assert nos.side == fix.SIDE_SELL
    assert nos.ord_type == fix.ORD_TYPE_LIMIT
    assert nos.price == 300.0
    assert nos.time_in_force == fix.TIF_GTC
    assert nos.text == "sma_v1"


def test_heartbeat_message():
    msg = heartbeat_message()
    assert msg.HasField("heartbeat")
    assert msg.heartbeat.test_req_id != ""
    assert msg.sender_comp_id == "QUANTBT"


def test_position_query_message():
    msg = position_query_message(strategy_id="sma_v1")
    assert msg.HasField("position_request")
    assert msg.position_request.account == "sma_v1"
    assert msg.position_request.pos_req_id != ""
