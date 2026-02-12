from quantbt.messaging.protocol import (
    Message,
    heartbeat_message,
    message_from_dict,
    new_order_message,
    position_query_message,
)


def test_message_roundtrip():
    msg = Message(msg_type="test", payload={"key": "value"})
    d = msg.to_dict()
    restored = Message.from_dict(d)
    assert restored.msg_type == "test"
    assert restored.payload["key"] == "value"
    assert restored.msg_id == msg.msg_id


def test_new_order_message():
    msg = new_order_message(
        cl_ord_id="test-001",
        instrument_dict={"symbol": "AAPL", "asset_class": "equity"},
        side="buy",
        quantity=100.0,
    )
    assert msg.msg_type == "new_order"
    assert msg.payload["cl_ord_id"] == "test-001"
    assert msg.payload["quantity"] == 100.0
    assert msg.payload["side"] == "buy"


def test_heartbeat_message():
    msg = heartbeat_message()
    assert msg.msg_type == "heartbeat"
    assert msg.payload == {}


def test_position_query_message():
    msg = position_query_message(strategy_id="sma_v1")
    assert msg.msg_type == "position_query"
    assert msg.payload["strategy_id"] == "sma_v1"


def test_message_from_dict():
    d = {
        "msg_type": "fill",
        "msg_id": "abc-123",
        "timestamp": "2024-01-01T00:00:00Z",
        "ref_msg_id": "orig-456",
        "payload": {"fill_price": 150.0},
    }
    msg = message_from_dict(d)
    assert msg.msg_type == "fill"
    assert msg.ref_msg_id == "orig-456"
    assert msg.payload["fill_price"] == 150.0
