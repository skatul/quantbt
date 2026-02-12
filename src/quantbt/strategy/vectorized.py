from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class VectorizedStrategy(ABC):
    """Vectorized strategy base. Operates on full DataFrames for speed."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Input: DataFrame with OHLCV columns.
        Output: DataFrame with added 'signal' column:
            1.0 = buy, -1.0 = sell, 0.0 = hold.
        """
        ...

    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position series. Default: hold until next signal."""
        signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(0)
        return signals
