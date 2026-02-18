from __future__ import annotations

import re
from dataclasses import dataclass, field

from quantbt.optimization.walk_forward import WalkForwardResult


@dataclass
class BiasReport:
    """Results from bias checks."""
    overfitting_detected: bool = False
    overfitting_details: str = ""
    parameter_stability: float = 0.0  # 0 = unstable, 1 = perfectly stable
    survivorship_warning: str = ""
    look_ahead_patterns: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return (
            self.overfitting_detected
            or bool(self.survivorship_warning)
            or bool(self.look_ahead_patterns)
        )


class BiasDetector:
    """Detects common backtesting biases."""

    def check_overfitting(self, wf_result: WalkForwardResult) -> BiasReport:
        """Compare in-sample vs out-of-sample performance across walk-forward windows.

        Checks:
        1. OOS Sharpe degradation: if average OOS objective is significantly lower
           than average IS objective, overfitting is likely.
        2. Parameter instability: if best params change across every window,
           the strategy may be curve-fitted.
        """
        report = BiasReport()
        if not wf_result.windows:
            return report

        train_objs = [w.train_objective for w in wf_result.windows]
        test_objs = [w.test_objective for w in wf_result.windows]

        avg_train = sum(train_objs) / len(train_objs)
        avg_test = sum(test_objs) / len(test_objs)

        # Check degradation
        if avg_train != 0:
            degradation = (avg_train - avg_test) / abs(avg_train) if avg_train != 0 else 0
        else:
            degradation = 0.0

        if degradation > 0.5:
            report.overfitting_detected = True
            report.overfitting_details = (
                f"Out-of-sample performance degraded by {degradation:.0%} vs in-sample. "
                f"Avg train objective: {avg_train:.4f}, avg test objective: {avg_test:.4f}"
            )

        # Check parameter stability
        all_params = wf_result.best_params_per_window
        if len(all_params) > 1:
            all_keys = set()
            for p in all_params:
                all_keys.update(p.keys())

            stable_count = 0
            total_count = 0
            for key in all_keys:
                values = [p.get(key) for p in all_params if key in p]
                if len(values) > 1:
                    total_count += 1
                    if len(set(str(v) for v in values)) == 1:
                        stable_count += 1

            report.parameter_stability = stable_count / total_count if total_count > 0 else 1.0

            if report.parameter_stability < 0.5:
                report.overfitting_detected = True
                report.overfitting_details += (
                    f" Parameter stability: {report.parameter_stability:.0%} "
                    f"(params change across {len(all_params)} windows)."
                )

        return report

    def check_survivorship_bias(
        self,
        symbols: list[str],
        start_year: int,
        min_symbols_per_year: int = 50,
    ) -> BiasReport:
        """Warn if trading a small universe over a long historical period.

        A small universe tested over many years likely suffers from survivorship
        bias — only stocks that survived to today are included.
        """
        report = BiasReport()
        import datetime as dt
        current_year = dt.date.today().year
        years = current_year - start_year

        if years > 5 and len(symbols) < min_symbols_per_year:
            report.survivorship_warning = (
                f"Testing {len(symbols)} symbols over {years} years. "
                f"Small universes over long periods are prone to survivorship bias. "
                f"Consider using a universe of at least {min_symbols_per_year} symbols "
                f"or reducing the test period."
            )
        return report

    def check_look_ahead(self, strategy_source: str) -> BiasReport:
        """Scan strategy source code for patterns that suggest look-ahead bias.

        Patterns checked:
        - shift(-N) — accessing future data
        - .iloc[-1] used in forward-looking context
        - Direct use of future dates/indices
        """
        report = BiasReport()

        patterns = [
            (r'\.shift\(\s*-\d+', "shift(-N) accesses future data"),
            (r'\.iloc\[\s*-\s*1\s*\].*shift', "iloc[-1] combined with shift may indicate look-ahead"),
            (r'\.pct_change\(\s*-', "pct_change with negative period accesses future data"),
            (r'\.diff\(\s*-', "diff with negative period accesses future data"),
            (r'\.rolling\(.*\)\.apply.*shift\s*\(\s*-', "rolling with negative shift"),
        ]

        for pattern, description in patterns:
            matches = re.findall(pattern, strategy_source)
            if matches:
                report.look_ahead_patterns.append(
                    f"{description} (found: {matches[0]})"
                )

        return report
