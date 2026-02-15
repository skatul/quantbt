#!/usr/bin/env python3
"""Run a local backtest using SimulatedBroker (no tradecore needed)."""

import argparse
from datetime import datetime

from quantbt.broker.simulated import SimulatedBroker
from quantbt.core.config import BacktestConfig
from quantbt.core.engine import BacktestEngine
from quantbt.core.logging import setup_logging
from quantbt.data.yahoo import YahooFinanceDataFeed
from quantbt.instrument.model import equity
from quantbt.portfolio.portfolio import Portfolio
from quantbt.strategy.examples.sma_crossover import SMACrossover


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantbt backtest")
    parser.add_argument("--config", default="config/default.toml", help="Path to TOML config file")
    parser.add_argument("--broker.commission_rate", type=float, dest="broker_commission_rate")
    parser.add_argument("--broker.slippage_bps", type=float, dest="broker_slippage_bps")
    parser.add_argument("--engine.initial_cash", type=float, dest="engine_initial_cash")
    parser.add_argument("--logging.level", dest="logging_level")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    # Build overrides from CLI args
    overrides: dict[str, object] = {}
    if args.broker_commission_rate is not None:
        overrides["broker.commission_rate"] = args.broker_commission_rate
    if args.broker_slippage_bps is not None:
        overrides["broker.slippage_bps"] = args.broker_slippage_bps
    if args.engine_initial_cash is not None:
        overrides["engine.initial_cash"] = args.engine_initial_cash
    if args.logging_level is not None:
        overrides["logging.level"] = args.logging_level

    cfg = BacktestConfig.load_with_overrides(args.config, **overrides)
    setup_logging(cfg.logging.level)

    # Date handling: CLI > config > defaults
    start_str = args.start or cfg.engine.start_date or "2024-01-01"
    end_str = args.end or cfg.engine.end_date or "2024-12-31"
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    # Strategy params from config
    sp = cfg.strategy.params
    fast_period = int(sp.get("fast_period", 10))
    slow_period = int(sp.get("slow_period", 30))
    trade_quantity = int(sp.get("trade_quantity", 100))

    instrument = equity(args.symbol, exchange="NASDAQ")
    portfolio = Portfolio(initial_cash=cfg.engine.initial_cash)
    broker = SimulatedBroker(
        commission_rate=cfg.broker.commission_rate,
        slippage_bps=cfg.broker.slippage_bps,
    )
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

    if engine.performance:
        print(f"\nPerformance:")
        print(f"  {engine.performance.summary()}")


if __name__ == "__main__":
    main()
