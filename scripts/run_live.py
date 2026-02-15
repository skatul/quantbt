#!/usr/bin/env python3
"""Run a backtest that sends orders to tradecore via ZMQ."""

import argparse
from datetime import datetime

from quantbt.broker.live import LiveBroker
from quantbt.core.config import BacktestConfig
from quantbt.core.engine import BacktestEngine
from quantbt.core.logging import setup_logging
from quantbt.data.yahoo import YahooFinanceDataFeed
from quantbt.instrument.model import equity
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.examples.sma_crossover import SMACrossover


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantbt via tradecore (ZMQ)")
    parser.add_argument("--config", default="config/default.toml", help="Path to TOML config file")
    parser.add_argument("--broker.commission_rate", type=float, dest="broker_commission_rate")
    parser.add_argument("--broker.zmq_address", dest="broker_zmq_address")
    parser.add_argument("--engine.initial_cash", type=float, dest="engine_initial_cash")
    parser.add_argument("--logging.level", dest="logging_level")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    overrides: dict[str, object] = {"broker.mode": "live"}
    if args.broker_commission_rate is not None:
        overrides["broker.commission_rate"] = args.broker_commission_rate
    if args.broker_zmq_address is not None:
        overrides["broker.zmq_address"] = args.broker_zmq_address
    if args.engine_initial_cash is not None:
        overrides["engine.initial_cash"] = args.engine_initial_cash
    if args.logging_level is not None:
        overrides["logging.level"] = args.logging_level

    cfg = BacktestConfig.load_with_overrides(args.config, **overrides)
    setup_logging(cfg.logging.level)

    start_str = args.start or cfg.engine.start_date or "2024-01-01"
    end_str = args.end or cfg.engine.end_date or "2024-12-31"
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    sp = cfg.strategy.params
    fast_period = int(sp.get("fast_period", 10))
    slow_period = int(sp.get("slow_period", 30))
    trade_quantity = int(sp.get("trade_quantity", 100))

    instrument = equity(args.symbol, exchange="NASDAQ")
    portfolio = Portfolio(initial_cash=cfg.engine.initial_cash)
    broker = LiveBroker(zmq_address=cfg.broker.zmq_address)
    strategy = SMACrossover(
        broker=broker,
        portfolio=portfolio,
        fast_period=fast_period,
        slow_period=slow_period,
        trade_quantity=trade_quantity,
    )
    data_feed = YahooFinanceDataFeed()

    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        broker=broker,
        instruments=[instrument],
        start=start,
        end=end,
    )

    print(f"Running SMA Crossover via tradecore on {instrument.symbol}...")
    print(f"Make sure tradecore is running on {cfg.broker.zmq_address}")
    result = engine.run()

    print(f"\nResults:")
    print(f"  Initial cash:     ${portfolio.initial_cash:,.2f}")
    print(f"  Final equity:     ${portfolio.total_equity:,.2f}")
    print(f"  Total return:     {portfolio.total_return:.2%}")
    print(f"  Realized PnL:     ${portfolio.total_realized_pnl:,.2f}")
    print(f"  Total commission: ${portfolio.total_commission:,.2f}")

    if engine.performance:
        print(f"\nPerformance:")
        print(f"  {engine.performance.summary()}")

    broker.close()


if __name__ == "__main__":
    main()
