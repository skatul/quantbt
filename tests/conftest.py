from datetime import datetime

import pandas as pd
import pytest

from quantbt.broker.simulated import SimulatedBroker
from quantbt.data.bar import Bar
from quantbt.instrument.model import equity
from quantbt.portfolio.portfolio import Portfolio


@pytest.fixture
def sample_bars():
    """Generate sample OHLCV bars for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    bars = []
    price = 100.0
    for i, dt in enumerate(dates):
        # Simulate trending then reversing price
        if i < 25:
            price += 0.5
        else:
            price -= 0.5
        bars.append(Bar(
            timestamp=dt.to_pydatetime(),
            open=price - 0.2,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=1_000_000.0,
            symbol="TEST",
        ))
    return bars


@pytest.fixture
def aapl_instrument():
    return equity("AAPL", exchange="NASDAQ")


@pytest.fixture
def portfolio():
    return Portfolio(initial_cash=100_000.0)


@pytest.fixture
def broker():
    return SimulatedBroker(commission_rate=0.001)
