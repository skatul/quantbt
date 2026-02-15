from datetime import datetime

from quantbt.broker.simulated import SimulatedBroker
from quantbt.data.bar import Bar
from quantbt.orders.model import Order, OrderType, Side
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.examples.sma_crossover import SMACrossover, SMACrossoverVectorized

import pandas as pd
import numpy as np


def test_sma_crossover_generates_trades(sample_bars, broker, portfolio):
    broker.on_fill(lambda order, fill: portfolio.update_position(
        order.symbol, order.side.value, fill.fill_quantity,
        fill.fill_price, fill.commission,
    ))

    strategy = SMACrossover(
        broker=broker,
        portfolio=portfolio,
        fast_period=5,
        slow_period=10,
        trade_quantity=10,
    )

    for bar in sample_bars:
        broker.set_current_bar(bar)
        strategy.on_bar(bar)

    # Should have generated at least one trade
    assert len(portfolio.positions) > 0


def test_vectorized_sma_crossover():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = np.concatenate([
        np.linspace(100, 120, 50),
        np.linspace(120, 95, 50),
    ])
    df = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": 1_000_000,
    }, index=dates)

    strategy = SMACrossoverVectorized(fast_period=5, slow_period=15)
    signals = strategy.generate_signals(df)

    assert "signal" in signals.columns
    # Should have at least one buy and one sell signal
    assert (signals["signal"] > 0).any()
    assert (signals["signal"] < 0).any()


def test_slippage_bps():
    """Verify slippage_bps adjusts fill prices correctly."""
    broker = SimulatedBroker(commission_rate=0.0, slippage_bps=10.0)
    fills = []
    broker.on_fill(lambda order, fill: fills.append((order, fill)))

    bar = Bar(
        timestamp=datetime(2024, 1, 1),
        open=100.0, high=102.0, low=99.0,
        close=100.0, volume=1_000_000, symbol="TEST",
    )
    broker.set_current_bar(bar)

    # Buy order: should fill at 100 * (1 + 10/10000) = 100.10
    buy_order = Order(
        cl_ord_id="slip-buy",
        symbol="TEST",
        side=Side.BUY,
        quantity=100.0,
        order_type=OrderType.MARKET,
    )
    broker.submit_order(buy_order)
    assert len(fills) == 1
    assert abs(fills[0][1].fill_price - 100.10) < 0.001

    # Sell order: should fill at 100 * (1 - 10/10000) = 99.90
    sell_order = Order(
        cl_ord_id="slip-sell",
        symbol="TEST",
        side=Side.SELL,
        quantity=100.0,
        order_type=OrderType.MARKET,
    )
    broker.submit_order(sell_order)
    assert len(fills) == 2
    assert abs(fills[1][1].fill_price - 99.90) < 0.001


def test_slippage_zero_by_default():
    """Default slippage is 0 — backward compatible."""
    broker = SimulatedBroker(commission_rate=0.0)
    fills = []
    broker.on_fill(lambda order, fill: fills.append(fill))

    bar = Bar(
        timestamp=datetime(2024, 1, 1),
        open=100.0, high=102.0, low=99.0,
        close=100.0, volume=1_000_000, symbol="TEST",
    )
    broker.set_current_bar(bar)

    order = Order(
        cl_ord_id="no-slip",
        symbol="TEST",
        side=Side.BUY,
        quantity=50.0,
        order_type=OrderType.MARKET,
    )
    broker.submit_order(order)
    assert len(fills) == 1
    assert fills[0].fill_price == 100.0
