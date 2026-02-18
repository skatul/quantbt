from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"
    FX = "fx"
    CRYPTO = "crypto"
    ETF = "etf"


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    asset_class: AssetClass
    exchange: str = ""
    currency: str = "USD"

    # Futures
    expiry: date | None = None
    contract_size: float = 1.0
    tick_size: float = 0.01

    # Options
    underlying: str | None = None
    strike: float | None = None
    option_type: str | None = None  # "call" or "put"
    expiration: date | None = None

    # FX / Crypto
    base_currency: str | None = None
    quote_currency: str | None = None
    pip_size: float | None = None

    # Trading hours: "24/7" for crypto, "regular" for equities, etc.
    trading_hours: str = "regular"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asset_class"] = self.asset_class.value
        for key in ("expiry", "expiration"):
            if d[key] is not None:
                d[key] = d[key].isoformat()
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> Instrument:
        d = d.copy()
        d["asset_class"] = AssetClass(d["asset_class"])
        for key in ("expiry", "expiration"):
            if key in d and d[key] is not None:
                d[key] = date.fromisoformat(d[key])
        return cls(**d)


def equity(symbol: str, exchange: str = "", currency: str = "USD") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.EQUITY,
                      exchange=exchange, currency=currency)


def future(symbol: str, expiry: date, exchange: str = "",
           contract_size: float = 1.0, currency: str = "USD") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.FUTURE,
                      exchange=exchange, expiry=expiry,
                      contract_size=contract_size, currency=currency)


def option(symbol: str, underlying: str, strike: float,
           option_type: str, expiration: date, exchange: str = "",
           currency: str = "USD") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.OPTION,
                      exchange=exchange, underlying=underlying,
                      strike=strike, option_type=option_type,
                      expiration=expiration, currency=currency)


def fx_pair(base: str, quote: str, pip_size: float = 0.0001) -> Instrument:
    return Instrument(symbol=f"{base}{quote}", asset_class=AssetClass.FX,
                      base_currency=base, quote_currency=quote,
                      pip_size=pip_size, currency=quote)


def crypto(symbol: str, exchange: str = "", currency: str = "USD") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.CRYPTO,
                      exchange=exchange, currency=currency,
                      trading_hours="24/7")


def etf(symbol: str, exchange: str = "", currency: str = "USD") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.ETF,
                      exchange=exchange, currency=currency)
