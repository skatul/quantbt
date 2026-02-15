"""Integration tests: quantbt <-> tradecore over ZMQ.

These tests start tradecore as a subprocess and send orders via ZMQ.
Requires tradecore binary at ../tradecore/build/tradecore.
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest

from quantbt.broker.live import LiveBroker
from quantbt.data.bar import Bar
from quantbt.messaging import fix_messages_pb2 as fix
from quantbt.messaging.protocol import (
    heartbeat_message,
    new_order_message,
    position_query_message,
)
from quantbt.messaging.zmq_client import ZmqClient
from quantbt.orders.model import Order, OrderStatus, OrderType, Side

TRADECORE_BINARY = Path(__file__).parent.parent.parent / "tradecore" / "build" / "tradecore"
ZMQ_PORT = 5557  # Use a non-default port to avoid conflicts
ZMQ_ADDRESS = f"tcp://127.0.0.1:{ZMQ_PORT}"


@pytest.fixture(scope="module")
def tradecore_server():
    """Start tradecore as a subprocess for the test module."""
    if not TRADECORE_BINARY.exists():
        pytest.skip(f"tradecore binary not found at {TRADECORE_BINARY}")

    proc = subprocess.Popen(
        [str(TRADECORE_BINARY), f"--bind=tcp://*:{ZMQ_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)  # Give it time to bind

    if proc.poll() is not None:
        pytest.fail(f"tradecore failed to start: {proc.stderr.read().decode()}")

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def zmq_client(tradecore_server):
    """Create a ZmqClient connected to the test server."""
    client = ZmqClient(address=ZMQ_ADDRESS, identity=f"test-{uuid4().hex[:8]}")
    time.sleep(0.1)  # Let connection establish with ROUTER
    yield client
    client.close()


class TestZmqIntegration:
    def test_single_market_order_fill(self, zmq_client):
        """Send a market order and verify we get an ExecutionReport fill back."""
        msg = new_order_message(
            cl_ord_id="int-001",
            instrument_dict={"symbol": "AAPL", "asset_class": "equity"},
            side="buy",
            quantity=100.0,
            market_price=185.50,
        )
        zmq_client.send(msg)

        response = zmq_client.recv(timeout_ms=2000)
        assert response is not None, "No response from tradecore"

        assert response.HasField("execution_report")
        er = response.execution_report
        assert er.cl_ord_id == "int-001"
        assert abs(er.last_px - 185.50) < 1.0  # Order book adds spread
        assert er.last_qty == 100.0
        assert er.ord_status == fix.ORD_STATUS_FILLED
        assert er.exec_type == fix.EXEC_TYPE_FILL

    def test_sell_order_fill(self, zmq_client):
        """Send a sell order and verify fill."""
        msg = new_order_message(
            cl_ord_id="int-002",
            instrument_dict={"symbol": "NVDA", "asset_class": "equity"},
            side="sell",
            quantity=50.0,
            market_price=190.00,
        )
        zmq_client.send(msg)

        response = zmq_client.recv(timeout_ms=2000)
        assert response is not None
        assert response.HasField("execution_report")
        er = response.execution_report
        assert abs(er.last_px - 190.00) < 1.0  # Order book adds spread
        assert er.last_qty == 50.0

    def test_multiple_orders_sequential(self, zmq_client):
        """Send multiple orders and verify each gets a fill."""
        for i in range(3):
            msg = new_order_message(
                cl_ord_id=f"int-multi-{i}",
                instrument_dict={"symbol": "MSFT", "asset_class": "equity"},
                side="buy",
                quantity=10.0,
                market_price=400.0 + i,
            )
            zmq_client.send(msg)

            response = zmq_client.recv(timeout_ms=2000)
            assert response is not None
            assert response.HasField("execution_report")
            assert response.execution_report.cl_ord_id == f"int-multi-{i}"

    def test_reject_invalid_quantity(self, zmq_client):
        """Order with zero quantity should be rejected."""
        msg = new_order_message(
            cl_ord_id="int-bad-001",
            instrument_dict={"symbol": "AAPL", "asset_class": "equity"},
            side="buy",
            quantity=0.0,
            market_price=185.0,
        )
        zmq_client.send(msg)

        response = zmq_client.recv(timeout_ms=2000)
        assert response is not None
        assert response.HasField("reject")
        assert response.reject.text != ""

    def test_heartbeat(self, zmq_client):
        """Heartbeat should get a heartbeat response."""
        msg = heartbeat_message()
        zmq_client.send(msg)

        response = zmq_client.recv(timeout_ms=2000)
        assert response is not None
        assert response.HasField("heartbeat")

    def test_position_query(self, zmq_client):
        """After orders, position query should return positions."""
        # First send an order to create a position
        msg = new_order_message(
            cl_ord_id="int-pos-001",
            instrument_dict={"symbol": "GOOG", "asset_class": "equity"},
            side="buy",
            quantity=25.0,
            market_price=175.0,
        )
        zmq_client.send(msg)
        response = zmq_client.recv(timeout_ms=2000)
        assert response is not None

        # Now query positions
        query = position_query_message(strategy_id="test")
        zmq_client.send(query)

        response = zmq_client.recv(timeout_ms=2000)
        assert response is not None
        assert response.HasField("position_report")
        pr = response.position_report
        assert len(pr.positions) > 0

        # Find our GOOG position
        goog_entries = [e for e in pr.positions if e.instrument.symbol == "GOOG"]
        assert len(goog_entries) == 1
        assert goog_entries[0].long_qty == 25.0


class TestLiveBrokerIntegration:
    def test_live_broker_submit_and_fill(self, tradecore_server):
        """Test LiveBroker end-to-end: submit order, get fill callback."""
        broker = LiveBroker(zmq_address=ZMQ_ADDRESS)
        fills_received = []

        def on_fill(order, fill):
            fills_received.append((order, fill))

        broker.on_fill(on_fill)

        # Set current bar so LiveBroker sends market price
        bar = Bar(
            timestamp=__import__("datetime").datetime(2024, 6, 1),
            open=150.0, high=152.0, low=149.0,
            close=151.50, volume=1_000_000, symbol="TSLA",
        )
        broker.set_current_bar(bar)

        order = Order(
            cl_ord_id="lb-001",
            symbol="TSLA",
            side=Side.BUY,
            quantity=50.0,
            order_type=OrderType.MARKET,
            strategy_id="test",
        )
        broker.submit_order(order)

        # Poll until we get a response
        for _ in range(20):
            broker.poll_responses(timeout_ms=100)
            if fills_received:
                break

        broker.close()

        assert len(fills_received) == 1
        fill_order, fill = fills_received[0]
        assert fill_order.cl_ord_id == "lb-001"
        assert abs(fill.fill_price - 151.50) < 1.0  # Order book adds spread
        assert fill.fill_quantity == 50.0
