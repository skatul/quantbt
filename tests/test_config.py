"""Tests for quantbt config loading."""

import tempfile
from pathlib import Path

from quantbt.core.config import BacktestConfig


def test_defaults():
    cfg = BacktestConfig.defaults()
    assert cfg.broker.mode == "simulated"
    assert cfg.broker.commission_rate == 0.001
    assert cfg.broker.slippage_bps == 0.0
    assert cfg.broker.zmq_address == "tcp://127.0.0.1:5555"
    assert cfg.engine.initial_cash == 100_000.0
    assert cfg.logging.level == "INFO"


def test_load_from_file():
    toml_content = b"""
[broker]
mode = "live"
commission_rate = 0.002
slippage_bps = 5.0
zmq_address = "tcp://127.0.0.1:6666"

[engine]
initial_cash = 50000.0
start_date = "2024-01-01"

[logging]
level = "DEBUG"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        path = f.name

    cfg = BacktestConfig.load(path)
    assert cfg.broker.mode == "live"
    assert cfg.broker.commission_rate == 0.002
    assert cfg.broker.slippage_bps == 5.0
    assert cfg.broker.zmq_address == "tcp://127.0.0.1:6666"
    assert cfg.engine.initial_cash == 50_000.0
    assert cfg.engine.start_date == "2024-01-01"
    assert cfg.logging.level == "DEBUG"
    # Unset values use defaults
    assert cfg.engine.end_date == ""

    Path(path).unlink()


def test_missing_file_returns_defaults():
    cfg = BacktestConfig.load("/nonexistent/path.toml")
    assert cfg.broker.mode == "simulated"
    assert cfg.broker.commission_rate == 0.001
    assert cfg.engine.initial_cash == 100_000.0


def test_load_with_overrides():
    toml_content = b"""
[broker]
commission_rate = 0.001

[engine]
initial_cash = 100000.0
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        path = f.name

    cfg = BacktestConfig.load_with_overrides(
        path,
        **{
            "broker.commission_rate": 0.005,
            "engine.initial_cash": 50000,
            "logging.level": "WARNING",
        },
    )
    assert cfg.broker.commission_rate == 0.005
    assert cfg.engine.initial_cash == 50_000.0
    assert cfg.logging.level == "WARNING"

    Path(path).unlink()


def test_partial_toml():
    toml_content = b"""
[strategy]
name = "my_strategy"

[strategy.params]
lookback = 20
threshold = 0.5
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        path = f.name

    cfg = BacktestConfig.load(path)
    assert cfg.strategy.name == "my_strategy"
    assert cfg.strategy.params["lookback"] == 20
    assert cfg.strategy.params["threshold"] == 0.5
    # Other sections use defaults
    assert cfg.broker.mode == "simulated"
    assert cfg.engine.initial_cash == 100_000.0

    Path(path).unlink()
