#!/usr/bin/env python3
"""Run a local backtest using SimulatedBroker (no tradecore needed)."""

from datetime import datetime

from quantbt.broker.simulated import SimulatedBroker
from quantbt.core.engine import BacktestEngine
from quantbt.data.yahoo import YahooFinanceDataFeed
from quantbt.instrument.model import equity
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.examples.sma_crossover import SMACrossover


def main() -> None:
    # Setup
    instrument = equity("AAPL", exchange="NASDAQ")
    portfolio = Portfolio(initial_cash=100_000.0)
    broker = SimulatedBroker(commission_rate=0.001)
    strategy = SMACrossover(
        broker=broker,
        portfolio=portfolio,
        fast_period=10,
        slow_period=30,
        trade_quantity=100,
    )
    data_feed = YahooFinanceDataFeed()

    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        broker=broker,
        instruments=[instrument],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )

    # Run
    print(f"Running SMA Crossover backtest on {instrument.symbol}...")
    result = engine.run()

    # Report
    print(f"\nResults:")
    print(f"  Initial cash:    ${portfolio.initial_cash:,.2f}")
    print(f"  Final equity:    ${portfolio.total_equity:,.2f}")
    print(f"  Total return:    {portfolio.total_return:.2%}")
    print(f"  Realized PnL:    ${portfolio.total_realized_pnl:,.2f}")
    print(f"  Total commission: ${portfolio.total_commission:,.2f}")
    print(f"  Cash remaining:  ${portfolio.cash:,.2f}")
    print(f"\nPositions:")
    for symbol, pos in portfolio.positions.items():
        print(f"  {symbol}: qty={pos.quantity}, avg_price=${pos.avg_price:.2f}, "
              f"realized_pnl=${pos.realized_pnl:.2f}")


if __name__ == "__main__":
    main()
