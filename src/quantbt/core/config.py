"""Configuration management for quantbt using TOML files + kwargs overrides."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import,no-redef]


@dataclass
class BrokerConfig:
    mode: str = "simulated"  # "simulated" or "live"
    commission_rate: float = 0.001
    slippage_bps: float = 0.0
    zmq_address: str = "tcp://127.0.0.1:5555"


@dataclass
class EngineConfig:
    initial_cash: float = 100_000.0
    start_date: str = ""
    end_date: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class StrategyConfig:
    name: str = ""
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @staticmethod
    def defaults() -> BacktestConfig:
        return BacktestConfig()

    @staticmethod
    def load(path: str | Path) -> BacktestConfig:
        """Load config from a TOML file. Missing file returns defaults."""
        cfg = BacktestConfig()
        p = Path(path)
        if not p.exists():
            return cfg

        with open(p, "rb") as f:
            data = tomllib.load(f)

        _apply_toml(cfg, data)
        return cfg

    @staticmethod
    def load_with_overrides(path: str | Path, **kwargs: object) -> BacktestConfig:
        """Load from TOML, then apply keyword overrides.

        Override keys use dot notation mapped to flat names:
          broker.commission_rate=0.002
          engine.initial_cash=50000
          logging.level=DEBUG
        """
        cfg = BacktestConfig.load(path)
        _apply_overrides(cfg, kwargs)
        return cfg


def _apply_toml(cfg: BacktestConfig, data: dict) -> None:
    if "broker" in data:
        b = data["broker"]
        if "mode" in b:
            cfg.broker.mode = str(b["mode"])
        if "commission_rate" in b:
            cfg.broker.commission_rate = float(b["commission_rate"])
        if "slippage_bps" in b:
            cfg.broker.slippage_bps = float(b["slippage_bps"])
        if "zmq_address" in b:
            cfg.broker.zmq_address = str(b["zmq_address"])

    if "engine" in data:
        e = data["engine"]
        if "initial_cash" in e:
            cfg.engine.initial_cash = float(e["initial_cash"])
        if "start_date" in e:
            cfg.engine.start_date = str(e["start_date"])
        if "end_date" in e:
            cfg.engine.end_date = str(e["end_date"])

    if "logging" in data:
        lg = data["logging"]
        if "level" in lg:
            cfg.logging.level = str(lg["level"])

    if "strategy" in data:
        s = data["strategy"]
        if "name" in s:
            cfg.strategy.name = str(s["name"])
        if "params" in s:
            cfg.strategy.params = dict(s["params"])


def _apply_overrides(cfg: BacktestConfig, overrides: dict[str, object]) -> None:
    mapping: dict[str, tuple[object, str]] = {
        "broker.mode": (cfg.broker, "mode"),
        "broker.commission_rate": (cfg.broker, "commission_rate"),
        "broker.slippage_bps": (cfg.broker, "slippage_bps"),
        "broker.zmq_address": (cfg.broker, "zmq_address"),
        "engine.initial_cash": (cfg.engine, "initial_cash"),
        "engine.start_date": (cfg.engine, "start_date"),
        "engine.end_date": (cfg.engine, "end_date"),
        "logging.level": (cfg.logging, "level"),
        "strategy.name": (cfg.strategy, "name"),
    }

    for key, value in overrides.items():
        if key in mapping:
            obj, attr = mapping[key]
            # Coerce to the same type as the default
            current = getattr(obj, attr)
            if isinstance(current, float):
                value = float(value)  # type: ignore[arg-type]
            elif isinstance(current, int):
                value = int(value)  # type: ignore[arg-type]
            else:
                value = str(value)
            setattr(obj, attr, value)
