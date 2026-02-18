"""Tests for streaming and vectorized indicators."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantbt.data.bar import Bar
from quantbt.indicators.moving_averages import SMA, EMA
from quantbt.indicators.oscillators import RSI, MACD, Stochastic
from quantbt.indicators.volatility import BollingerBands, ATR
from quantbt.indicators.volume import VWAP, OBV
from quantbt.indicators import vectorized as vec


# --------------- SMA ---------------

class TestSMA:
    def test_not_ready_before_period(self):
        sma = SMA(3)
        assert sma.update(1.0) is None
        assert sma.update(2.0) is None
        assert not sma.ready

    def test_first_value(self):
        sma = SMA(3)
        sma.update(1.0)
        sma.update(2.0)
        result = sma.update(3.0)
        assert result == pytest.approx(2.0)
        assert sma.ready
        assert sma.value == pytest.approx(2.0)

    def test_rolling(self):
        sma = SMA(3)
        for v in [1.0, 2.0, 3.0]:
            sma.update(v)
        # Window: [2, 3, 4]
        result = sma.update(4.0)
        assert result == pytest.approx(3.0)

    def test_reset(self):
        sma = SMA(3)
        for v in [1.0, 2.0, 3.0]:
            sma.update(v)
        assert sma.ready
        sma.reset()
        assert not sma.ready
        assert sma.value is None

    def test_known_values(self):
        """SMA(5) on [10, 11, 12, 13, 14, 15] -> first = 12.0, second = 13.0."""
        sma = SMA(5)
        values = [10, 11, 12, 13, 14, 15]
        results = [sma.update(v) for v in values]
        assert results[:4] == [None, None, None, None]
        assert results[4] == pytest.approx(12.0)
        assert results[5] == pytest.approx(13.0)


# --------------- EMA ---------------

class TestEMA:
    def test_first_value_is_sma(self):
        e = EMA(3)
        e.update(2.0)
        e.update(4.0)
        result = e.update(6.0)
        assert result == pytest.approx(4.0)  # SMA(2,4,6) = 4.0

    def test_subsequent_values(self):
        e = EMA(3)
        for v in [2.0, 4.0, 6.0]:
            e.update(v)
        # k = 2/(3+1) = 0.5; EMA = 4.0 + 0.5*(8.0 - 4.0) = 6.0
        result = e.update(8.0)
        assert result == pytest.approx(6.0)

    def test_reset(self):
        e = EMA(3)
        for v in [1.0, 2.0, 3.0]:
            e.update(v)
        e.reset()
        assert not e.ready
        assert e.value is None


# --------------- RSI ---------------

class TestRSI:
    def test_all_gains(self):
        r = RSI(period=5)
        # Feed: 100, 101, 102, 103, 104, 105 (all gains)
        for v in [100, 101, 102, 103, 104, 105]:
            r.update(v)
        assert r.ready
        assert r.value == pytest.approx(100.0)

    def test_all_losses(self):
        r = RSI(period=5)
        for v in [105, 104, 103, 102, 101, 100]:
            r.update(v)
        assert r.ready
        assert r.value == pytest.approx(0.0)

    def test_mixed(self):
        r = RSI(period=3)
        values = [44, 44.34, 44.09, 43.61]
        for v in values:
            r.update(v)
        assert r.ready
        assert 0.0 <= r.value <= 100.0

    def test_not_ready_before_period(self):
        r = RSI(period=14)
        for v in range(10):
            r.update(float(v))
        assert not r.ready


# --------------- MACD ---------------

class TestMACD:
    def test_needs_slow_period_data(self):
        m = MACD(fast=3, slow=5, signal=3)
        for v in range(4):
            m.update(float(v))
        assert not m.ready

    def test_produces_value_after_slow_period(self):
        m = MACD(fast=3, slow=5, signal=3)
        for v in range(10):
            m.update(float(v))
        assert m.ready
        assert m.value is not None

    def test_signal_line_after_signal_period(self):
        m = MACD(fast=3, slow=5, signal=2)
        results = []
        for v in range(10):
            results.append(m.update(float(v)))
        assert m.signal_line is not None
        assert m.histogram is not None


# --------------- Stochastic ---------------

class TestStochastic:
    def test_not_ready_before_k_period(self):
        s = Stochastic(k_period=5, d_period=3)
        for i in range(4):
            s.update_hlc(10.0 + i, 9.0 + i, 9.5 + i)
        assert not s.ready

    def test_known_value(self):
        s = Stochastic(k_period=3, d_period=2)
        # H/L/C: (12,8,10), (14,9,13), (13,10,12)
        s.update_hlc(12, 8, 10)
        s.update_hlc(14, 9, 13)
        result = s.update_hlc(13, 10, 12)
        # Highest high = 14, lowest low = 8, %K = 100*(12-8)/(14-8) = 66.67
        assert result == pytest.approx(100.0 * 4 / 6, rel=1e-4)

    def test_d_line(self):
        s = Stochastic(k_period=3, d_period=2)
        for i in range(5):
            s.update_hlc(10.0 + i, 8.0 + i, 9.0 + i)
        assert s.d is not None


# --------------- BollingerBands ---------------

class TestBollingerBands:
    def test_not_ready_before_period(self):
        bb = BollingerBands(period=3, num_std=2.0)
        bb.update(10.0)
        bb.update(11.0)
        assert not bb.ready

    def test_constant_values_no_bands(self):
        bb = BollingerBands(period=3, num_std=2.0)
        for _ in range(3):
            bb.update(10.0)
        assert bb.middle == pytest.approx(10.0)
        assert bb.upper == pytest.approx(10.0)
        assert bb.lower == pytest.approx(10.0)

    def test_spread(self):
        bb = BollingerBands(period=3, num_std=1.0)
        bb.update(10.0)
        bb.update(20.0)
        bb.update(30.0)
        assert bb.middle == pytest.approx(20.0)
        assert bb.upper > bb.middle
        assert bb.lower < bb.middle


# --------------- ATR ---------------

class TestATR:
    def test_close_only(self):
        a = ATR(period=3)
        # Prices: 100, 102, 99, 101, 103
        vals = [100, 102, 99, 101, 103]
        results = [a.update(v) for v in vals]
        assert results[0] is None  # first value, no prev
        assert results[1] is None  # count 1 < period
        assert results[2] is None  # count 2 < period
        assert results[3] is not None  # count 3 == period
        assert a.ready

    def test_bar_method(self):
        from datetime import datetime
        a = ATR(period=2)
        b1 = Bar(datetime(2024, 1, 1), 10.0, 12.0, 9.0, 11.0, 100.0, "T")
        b2 = Bar(datetime(2024, 1, 2), 11.0, 14.0, 10.0, 13.0, 100.0, "T")
        b3 = Bar(datetime(2024, 1, 3), 13.0, 15.0, 12.0, 14.0, 100.0, "T")
        a.update_bar(b1)  # sets prev_close, no TR
        a.update_bar(b2)  # TR = max(14-10, |14-11|, |10-11|) = 4.0
        result = a.update_bar(b3)  # TR = max(15-12, |15-13|, |12-13|) = 3.0
        # ATR = (4.0 + 3.0) / 2 = 3.5
        assert result == pytest.approx(3.5)


# --------------- VWAP ---------------

class TestVWAP:
    def test_single_bar(self):
        from datetime import datetime
        v = VWAP()
        b = Bar(datetime(2024, 1, 1), 10.0, 12.0, 9.0, 11.0, 1000.0, "T")
        result = v.update_bar(b)
        typical = (12.0 + 9.0 + 11.0) / 3.0
        assert result == pytest.approx(typical)

    def test_cumulative(self):
        v = VWAP()
        v.update_pv(10.0, 100.0)  # cum_pv = 1000, cum_vol = 100
        result = v.update_pv(20.0, 200.0)  # cum_pv = 5000, cum_vol = 300
        assert result == pytest.approx(5000.0 / 300.0)

    def test_reset(self):
        v = VWAP()
        v.update_pv(10.0, 100.0)
        v.reset()
        assert not v.ready


# --------------- OBV ---------------

class TestOBV:
    def test_up_move(self):
        o = OBV()
        o.update_cv(10.0, 100.0)
        result = o.update_cv(11.0, 200.0)
        assert result == pytest.approx(300.0)

    def test_down_move(self):
        o = OBV()
        o.update_cv(10.0, 100.0)
        result = o.update_cv(9.0, 200.0)
        assert result == pytest.approx(-100.0)

    def test_flat_move(self):
        o = OBV()
        o.update_cv(10.0, 100.0)
        result = o.update_cv(10.0, 200.0)
        assert result == pytest.approx(100.0)


# --------------- Vectorized functions ---------------

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


class TestVectorized:
    def test_sma(self, sample_df):
        result = vec.sma(sample_df["close"], 10)
        assert len(result) == len(sample_df)
        assert result.isna().sum() == 9  # period - 1 NaN

    def test_ema(self, sample_df):
        result = vec.ema(sample_df["close"], 10)
        assert len(result) == len(sample_df)
        assert not result.isna().any()

    def test_rsi(self, sample_df):
        result = vec.rsi(sample_df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd(self, sample_df):
        result = vec.macd(sample_df["close"])
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_bollinger_bands(self, sample_df):
        result = vec.bollinger_bands(sample_df["close"], 20)
        assert "middle" in result.columns
        assert "upper" in result.columns
        assert "lower" in result.columns
        valid = result.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["lower"] <= valid["middle"]).all()

    def test_atr(self, sample_df):
        result = vec.atr(sample_df, 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_vwap(self, sample_df):
        result = vec.vwap(sample_df)
        assert len(result) == len(sample_df)
        assert not result.isna().any()

    def test_obv(self, sample_df):
        result = vec.obv(sample_df)
        assert len(result) == len(sample_df)

    def test_stochastic(self, sample_df):
        result = vec.stochastic(sample_df)
        assert "k" in result.columns
        assert "d" in result.columns
        valid = result.dropna()
        assert (valid["k"] >= 0).all() and (valid["k"] <= 100).all()


# --------------- update_bar convenience ---------------

class TestUpdateBar:
    def test_sma_update_bar(self):
        from datetime import datetime
        sma = SMA(2)
        b1 = Bar(datetime(2024, 1, 1), 10.0, 12.0, 9.0, 11.0, 100.0, "T")
        b2 = Bar(datetime(2024, 1, 2), 11.0, 13.0, 10.0, 12.0, 100.0, "T")
        sma.update_bar(b1)
        result = sma.update_bar(b2)
        assert result == pytest.approx(11.5)
