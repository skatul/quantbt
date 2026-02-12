from quantbt.portfolio.portfolio import Portfolio, Position


def test_position_buy():
    pos = Position(symbol="AAPL")
    pos.apply_fill("buy", 100, 150.0)
    assert pos.quantity == 100
    assert pos.avg_price == 150.0
    assert pos.cost_basis == 15_000.0


def test_position_buy_sell_pnl():
    pos = Position(symbol="AAPL")
    pos.apply_fill("buy", 100, 150.0)
    pos.apply_fill("sell", 100, 160.0)
    assert pos.quantity == 0
    assert pos.realized_pnl == 1_000.0  # (160-150) * 100
    assert pos.is_flat


def test_position_short():
    pos = Position(symbol="AAPL")
    pos.apply_fill("sell", 100, 150.0)
    assert pos.quantity == -100
    pos.apply_fill("buy", 100, 140.0)
    assert pos.quantity == 0
    assert pos.realized_pnl == 1_000.0  # (150-140) * 100


def test_portfolio_cash_tracking():
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.update_position("AAPL", "buy", 100, 150.0, commission=10.0)
    assert portfolio.cash == 100_000.0 - 15_000.0 - 10.0
    assert portfolio.total_commission == 10.0


def test_portfolio_total_equity():
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.update_position("AAPL", "buy", 100, 150.0)
    # equity = cash + position value
    assert portfolio.total_equity == 100_000.0  # cash decreased by 15000, position worth 15000


def test_portfolio_total_return():
    portfolio = Portfolio(initial_cash=100_000.0)
    assert portfolio.total_return == 0.0
