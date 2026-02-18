"""Tests for Crypto/ETF instruments, CsvDataFeed, and periods_per_year."""

from __future__ import annotations

import tempfile
from datetime import datetime

import pandas as pd
import pytest

from quantbt.data.csv_feed import CsvDataFeed
from quantbt.instrument.model import AssetClass, crypto, etf, Instrument
from quantbt.portfolio.metrics import PerformanceTracker


class TestCryptoETFInstruments:
    def test_crypto_factory(self):
        btc = crypto("BTC/USD", exchange="binance")
        assert btc.asset_class == AssetClass.CRYPTO
        assert btc.trading_hours == "24/7"
        assert btc.currency == "USD"

    def test_etf_factory(self):
        spy = etf("SPY", exchange="NYSE")
        assert spy.asset_class == AssetClass.ETF
        assert spy.trading_hours == "regular"

    def test_crypto_to_dict_roundtrip(self):
        btc = crypto("BTC/USD", exchange="binance")
        d = btc.to_dict()
        assert d["asset_class"] == "crypto"
        assert d["trading_hours"] == "24/7"
        restored = Instrument.from_dict(d)
        assert restored.asset_class == AssetClass.CRYPTO

    def test_etf_to_dict_roundtrip(self):
        spy = etf("SPY", exchange="NYSE")
        d = spy.to_dict()
        assert d["asset_class"] == "etf"
        restored = Instrument.from_dict(d)
        assert restored.asset_class == AssetClass.ETF


class TestCsvDataFeed:
    def _write_csv(self, tmp, rows):
        lines = ["date,open,high,low,close,volume"]
        for r in rows:
            lines.append(",".join(str(v) for v in r))
        path = tmp / "test.csv"
        path.write_text("\n".join(lines))
        return path

    def test_fetch(self, tmp_path):
        rows = [
            ("2024-01-01", 100, 105, 95, 102, 1000),
            ("2024-01-02", 102, 108, 100, 106, 1200),
            ("2024-01-03", 106, 110, 104, 109, 1100),
        ]
        path = self._write_csv(tmp_path, rows)
        feed = CsvDataFeed(path)
        inst = crypto("BTC/USD")
        df = feed.fetch(inst, datetime(2024, 1, 1), datetime(2024, 1, 3))
        assert len(df) == 3
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_fetch_filters_date_range(self, tmp_path):
        rows = [
            ("2024-01-01", 100, 105, 95, 102, 1000),
            ("2024-01-02", 102, 108, 100, 106, 1200),
            ("2024-01-03", 106, 110, 104, 109, 1100),
            ("2024-01-04", 109, 112, 107, 111, 900),
        ]
        path = self._write_csv(tmp_path, rows)
        feed = CsvDataFeed(path)
        inst = crypto("BTC/USD")
        df = feed.fetch(inst, datetime(2024, 1, 2), datetime(2024, 1, 3))
        assert len(df) == 2

    def test_iter_bars(self, tmp_path):
        rows = [
            ("2024-01-01", 100, 105, 95, 102, 1000),
            ("2024-01-02", 102, 108, 100, 106, 1200),
        ]
        path = self._write_csv(tmp_path, rows)
        feed = CsvDataFeed(path)
        inst = crypto("BTC/USD")
        bars = list(feed.iter_bars(inst, datetime(2024, 1, 1), datetime(2024, 1, 2)))
        assert len(bars) == 2
        assert bars[0].symbol == "BTC/USD"
        assert bars[0].close == 102

    def test_missing_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("date,open,close\n2024-01-01,100,102\n")
        feed = CsvDataFeed(path)
        inst = crypto("BTC/USD")
        with pytest.raises(ValueError, match="missing required columns"):
            feed.fetch(inst, datetime(2024, 1, 1), datetime(2024, 1, 1))


class TestPeriodsPerYear:
    def test_default_252(self):
        tracker = PerformanceTracker(initial_cash=100_000.0)
        assert tracker.periods_per_year == 252.0

    def test_crypto_365(self):
        tracker = PerformanceTracker(initial_cash=100_000.0, periods_per_year=365.0)
        assert tracker.periods_per_year == 365.0

    def test_sharpe_uses_instance_periods(self):
        t1 = PerformanceTracker(initial_cash=100.0, periods_per_year=252.0)
        t2 = PerformanceTracker(initial_cash=100.0, periods_per_year=365.0)
        # Feed identical equity points
        equity = 100.0
        for i in range(20):
            equity += 0.1
            ts = datetime(2024, 1, 1 + i)
            t1.record(ts, equity, equity, 0.0)
            t2.record(ts, equity, equity, 0.0)
        # Sharpe with 365 periods should differ from 252
        s252 = t1.sharpe_ratio()
        s365 = t2.sharpe_ratio()
        assert s252 != pytest.approx(s365, abs=0.01)
