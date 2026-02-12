#!/usr/bin/env python3
"""Run a backtest that sends orders to tradecore via ZMQ."""

from datetime import datetime

from quantbt.broker.live import LiveBroker
from quantbt.core.engine import BacktestEngine
from quantbt.data.yahoo import YahooFinanceDataFeed
from quantbt.instrument.model import equity
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.examples.sma_crossover import SMACrossover


def main() -> None:
    instrument = equity("AAPL", exchange="NASDAQ")
    portfolio = Portfolio(initial_cash=100_000.0)
    broker = LiveBroker(zmq_address="tcp://127.0.0.1:5555")
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

    print(f"Running SMA Crossover via tradecore on {instrument.symbol}...")
    print("Make sure tradecore is running on tcp://127.0.0.1:5555")
    result = engine.run()

    print(f"\nResults:")
    print(f"  Initial cash:     ${portfolio.initial_cash:,.2f}")
    print(f"  Final equity:     ${portfolio.total_equity:,.2f}")
    print(f"  Total return:     {portfolio.total_return:.2%}")
    print(f"  Realized PnL:     ${portfolio.total_realized_pnl:,.2f}")
    print(f"  Total commission: ${portfolio.total_commission:,.2f}")

    broker.close()


if __name__ == "__main__":
    main()
