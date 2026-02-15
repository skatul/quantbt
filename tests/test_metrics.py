from datetime import datetime, timedelta

from quantbt.portfolio.metrics import PerformanceTracker
from quantbt.portfolio.portfolio import Portfolio


def test_tracker_record_and_total_return():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    base = datetime(2024, 1, 1)

    tracker.record(base, 100_000.0, 100_000.0, 0.0)
    tracker.record(base + timedelta(days=1), 101_000.0, 90_000.0, 11_000.0)
    tracker.record(base + timedelta(days=2), 102_000.0, 90_000.0, 12_000.0)

    assert len(tracker.equity_curve) == 3
    assert abs(tracker.total_return() - 0.02) < 0.001


def test_max_drawdown():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    base = datetime(2024, 1, 1)

    tracker.record(base, 100_000.0, 100_000.0, 0.0)
    tracker.record(base + timedelta(days=1), 110_000.0, 100_000.0, 10_000.0)
    tracker.record(base + timedelta(days=2), 99_000.0, 99_000.0, 0.0)  # drawdown
    tracker.record(base + timedelta(days=3), 105_000.0, 100_000.0, 5_000.0)

    dd = tracker.max_drawdown()
    # Peak was 110k, trough was 99k -> dd = (110-99)/110 = 0.1
    assert abs(dd - 0.1) < 0.001


def test_sharpe_ratio():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    base = datetime(2024, 1, 1)

    # Steady growth: +1% per day
    equity = 100_000.0
    for i in range(30):
        equity *= 1.01
        tracker.record(base + timedelta(days=i), equity, equity, 0.0)

    sharpe = tracker.sharpe_ratio()
    # With consistent positive returns, Sharpe should be very high
    assert sharpe > 5.0


def test_sortino_ratio():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    base = datetime(2024, 1, 1)

    # Mostly up, one down day
    equities = [100_000, 101_000, 102_000, 100_500, 103_000, 104_000]
    for i, eq in enumerate(equities):
        tracker.record(base + timedelta(days=i), eq, eq, 0.0)

    sortino = tracker.sortino_ratio()
    assert sortino > 0


def test_win_rate():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    base = datetime(2024, 1, 1)

    equities = [100_000, 101_000, 99_000, 102_000, 100_500]
    for i, eq in enumerate(equities):
        tracker.record(base + timedelta(days=i), eq, eq, 0.0)

    wr = tracker.win_rate()
    # Returns: 0%, +1%, -2%, +3%, -1.5%  -> 2 wins out of 5 = 0.4
    assert abs(wr - 0.4) < 0.01


def test_summary():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    base = datetime(2024, 1, 1)

    for i in range(10):
        tracker.record(base + timedelta(days=i), 100_000.0 + i * 100, 100_000.0, i * 100)

    s = tracker.summary()
    assert "total_return" in s
    assert "max_drawdown" in s
    assert "sharpe_ratio" in s
    assert "sortino_ratio" in s
    assert "win_rate" in s
    assert "num_bars" in s
    assert s["num_bars"] == 10


def test_mark_to_market():
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.update_position("AAPL", "buy", 100, 150.0)

    # With avg_price, equity = cash + 100*150 = 85000 + 15000 = 100000
    assert portfolio.total_equity == 100_000.0

    # With MTM at 160, equity = 85000 + 100*160 = 101000
    mtm = portfolio.mark_to_market({"AAPL": 160.0})
    assert mtm == 101_000.0

    # Without AAPL in prices, falls back to avg_price
    mtm_fallback = portfolio.mark_to_market({})
    assert mtm_fallback == 100_000.0


def test_empty_tracker():
    tracker = PerformanceTracker(initial_cash=100_000.0)
    assert tracker.total_return() == 0.0
    assert tracker.max_drawdown() == 0.0
    assert tracker.sharpe_ratio() == 0.0
    assert tracker.sortino_ratio() == 0.0
    assert tracker.win_rate() == 0.0
