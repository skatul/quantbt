from quantbt.data.bar import Bar
from datetime import datetime


def test_bar_creation():
    bar = Bar(
        timestamp=datetime(2024, 1, 1),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1_000_000,
        symbol="AAPL",
    )
    assert bar.symbol == "AAPL"
    assert bar.close == 103.0
    assert bar.high == 105.0


def test_bar_is_frozen():
    bar = Bar(
        timestamp=datetime(2024, 1, 1),
        open=100.0, high=105.0, low=99.0,
        close=103.0, volume=1_000_000, symbol="AAPL",
    )
    try:
        bar.close = 200.0  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass
