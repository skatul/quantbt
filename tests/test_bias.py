"""Tests for bias detection: overfitting, survivorship, look-ahead."""

from __future__ import annotations

from datetime import datetime

import pytest

from quantbt.optimization.bias import BiasDetector, BiasReport
from quantbt.optimization.walk_forward import WalkForwardResult, WindowResult


# ---------- Helpers ----------

def _make_window(
    index: int,
    train_obj: float,
    test_obj: float,
    params: dict | None = None,
) -> WindowResult:
    return WindowResult(
        window_index=index,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 3, 1),
        test_start=datetime(2024, 3, 1),
        test_end=datetime(2024, 4, 1),
        best_params=params or {"fast": 10, "slow": 30},
        train_metrics={"sharpe_ratio": train_obj, "total_return": 0.1},
        test_metrics={"sharpe_ratio": test_obj, "total_return": 0.02},
        train_objective=train_obj,
        test_objective=test_obj,
    )


# ---------- Overfitting Detection ----------

class TestOverfittingDetection:
    def test_no_overfitting_when_oos_similar(self):
        detector = BiasDetector()
        wf = WalkForwardResult(windows=[
            _make_window(0, train_obj=1.5, test_obj=1.3),
            _make_window(1, train_obj=1.4, test_obj=1.2),
        ])
        report = detector.check_overfitting(wf)
        assert not report.overfitting_detected

    def test_overfitting_detected_on_degradation(self):
        detector = BiasDetector()
        wf = WalkForwardResult(windows=[
            _make_window(0, train_obj=2.0, test_obj=0.1),
            _make_window(1, train_obj=2.5, test_obj=-0.5),
        ])
        report = detector.check_overfitting(wf)
        assert report.overfitting_detected
        assert "degraded" in report.overfitting_details.lower()

    def test_parameter_instability_flagged(self):
        detector = BiasDetector()
        wf = WalkForwardResult(windows=[
            _make_window(0, train_obj=1.5, test_obj=1.3, params={"fast": 5, "slow": 20}),
            _make_window(1, train_obj=1.4, test_obj=1.2, params={"fast": 15, "slow": 50}),
            _make_window(2, train_obj=1.6, test_obj=1.1, params={"fast": 8, "slow": 35}),
        ])
        report = detector.check_overfitting(wf)
        assert report.parameter_stability < 0.5

    def test_stable_parameters(self):
        detector = BiasDetector()
        wf = WalkForwardResult(windows=[
            _make_window(0, train_obj=1.5, test_obj=1.3, params={"fast": 10, "slow": 30}),
            _make_window(1, train_obj=1.4, test_obj=1.2, params={"fast": 10, "slow": 30}),
        ])
        report = detector.check_overfitting(wf)
        assert report.parameter_stability == 1.0

    def test_empty_result(self):
        detector = BiasDetector()
        wf = WalkForwardResult(windows=[])
        report = detector.check_overfitting(wf)
        assert not report.overfitting_detected


# ---------- Survivorship Bias ----------

class TestSurvivorshipBias:
    def test_small_universe_long_period_warns(self):
        detector = BiasDetector()
        report = detector.check_survivorship_bias(
            symbols=["AAPL", "MSFT", "GOOG"],
            start_year=2010,
        )
        assert report.survivorship_warning
        assert "survivorship" in report.survivorship_warning.lower()

    def test_large_universe_no_warning(self):
        detector = BiasDetector()
        symbols = [f"SYM{i}" for i in range(100)]
        report = detector.check_survivorship_bias(symbols=symbols, start_year=2010)
        assert not report.survivorship_warning

    def test_short_period_no_warning(self):
        detector = BiasDetector()
        report = detector.check_survivorship_bias(
            symbols=["AAPL", "MSFT"],
            start_year=2023,
        )
        assert not report.survivorship_warning

    def test_custom_min_symbols(self):
        detector = BiasDetector()
        symbols = [f"SYM{i}" for i in range(20)]
        report = detector.check_survivorship_bias(
            symbols=symbols, start_year=2010, min_symbols_per_year=25,
        )
        assert report.survivorship_warning


# ---------- Look-Ahead Bias ----------

class TestLookAheadBias:
    def test_shift_negative_detected(self):
        detector = BiasDetector()
        code = '''
        def generate_signals(self, data):
            data["future_return"] = data["close"].shift(-1)
            return data
        '''
        report = detector.check_look_ahead(code)
        assert len(report.look_ahead_patterns) > 0
        assert "shift" in report.look_ahead_patterns[0].lower()

    def test_diff_negative_detected(self):
        detector = BiasDetector()
        code = 'df["x"] = df["close"].diff(-5)'
        report = detector.check_look_ahead(code)
        assert len(report.look_ahead_patterns) > 0

    def test_pct_change_negative_detected(self):
        detector = BiasDetector()
        code = 'df["fwd"] = df["close"].pct_change(-1)'
        report = detector.check_look_ahead(code)
        assert len(report.look_ahead_patterns) > 0

    def test_clean_code_no_warnings(self):
        detector = BiasDetector()
        code = '''
        def generate_signals(self, data):
            data["sma"] = data["close"].rolling(20).mean()
            data["signal"] = data["close"] > data["sma"]
            return data
        '''
        report = detector.check_look_ahead(code)
        assert len(report.look_ahead_patterns) == 0

    def test_positive_shift_not_flagged(self):
        detector = BiasDetector()
        code = 'df["prev"] = df["close"].shift(1)'
        report = detector.check_look_ahead(code)
        assert len(report.look_ahead_patterns) == 0


# ---------- BiasReport ----------

class TestBiasReport:
    def test_has_warnings_false_when_clean(self):
        report = BiasReport()
        assert not report.has_warnings

    def test_has_warnings_on_overfitting(self):
        report = BiasReport(overfitting_detected=True)
        assert report.has_warnings

    def test_has_warnings_on_survivorship(self):
        report = BiasReport(survivorship_warning="Some warning")
        assert report.has_warnings

    def test_has_warnings_on_look_ahead(self):
        report = BiasReport(look_ahead_patterns=["shift(-1)"])
        assert report.has_warnings
