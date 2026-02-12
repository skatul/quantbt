from datetime import date

from quantbt.instrument.model import (
    AssetClass,
    Instrument,
    equity,
    future,
    fx_pair,
    option,
)


def test_equity_factory():
    inst = equity("AAPL", exchange="NASDAQ")
    assert inst.symbol == "AAPL"
    assert inst.asset_class == AssetClass.EQUITY
    assert inst.exchange == "NASDAQ"
    assert inst.currency == "USD"


def test_future_factory():
    inst = future("ES", expiry=date(2024, 12, 20), exchange="CME", contract_size=50.0)
    assert inst.symbol == "ES"
    assert inst.asset_class == AssetClass.FUTURE
    assert inst.expiry == date(2024, 12, 20)
    assert inst.contract_size == 50.0


def test_option_factory():
    inst = option("AAPL240119C00190000", "AAPL", 190.0, "call", date(2024, 1, 19))
    assert inst.asset_class == AssetClass.OPTION
    assert inst.underlying == "AAPL"
    assert inst.strike == 190.0
    assert inst.option_type == "call"


def test_fx_pair_factory():
    inst = fx_pair("EUR", "USD")
    assert inst.symbol == "EURUSD"
    assert inst.asset_class == AssetClass.FX
    assert inst.base_currency == "EUR"
    assert inst.quote_currency == "USD"
    assert inst.pip_size == 0.0001


def test_instrument_json_roundtrip():
    inst = equity("AAPL", exchange="NASDAQ")
    d = inst.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["asset_class"] == "equity"

    restored = Instrument.from_dict(d)
    assert restored == inst


def test_future_json_roundtrip():
    inst = future("ES", expiry=date(2024, 12, 20), exchange="CME", contract_size=50.0)
    d = inst.to_dict()
    restored = Instrument.from_dict(d)
    assert restored == inst
    assert restored.expiry == date(2024, 12, 20)


def test_to_dict_strips_nones():
    inst = equity("AAPL")
    d = inst.to_dict()
    assert "expiry" not in d
    assert "underlying" not in d
    assert "base_currency" not in d
