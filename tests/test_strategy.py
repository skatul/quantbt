from quantbt.broker.simulated import SimulatedBroker
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
