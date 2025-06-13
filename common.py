"""
Common utility functions shared across indicator modules.

This module centralises frequently used simple computations such as moving
averages and rolling high/low helpers so they are defined in a single place.
Keeping these generic helpers in one file avoids duplicated code across
`basic_trend.py`, `breakout.py`, and other modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "wma",
    "rolling_high",
    "rolling_low",
]

# ---------------------------------------------------------------------------
# Moving average helpers
# ---------------------------------------------------------------------------

def sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average (SMA)."""
    return prices.rolling(window=period, min_periods=1).mean()


def ema(prices: pd.Series, period: int, *, alpha: float | None = None) -> pd.Series:
    """Exponential Moving Average (EMA).

    Parameters
    ----------
    prices : pd.Series
        Series of prices.
    period : int
        Look-back window length.
    alpha : float | None, default None
        Smoothing factor.  If *None* it is calculated as ``2 / (period + 1)``.
    """
    if alpha is None:
        alpha = 2.0 / (period + 1)
    return prices.ewm(alpha=alpha, adjust=False).mean()


def wma(prices: pd.Series, period: int) -> pd.Series:
    """Linear Weighted Moving Average (WMA)."""

    def _weighted_mean(x: np.ndarray) -> float:
        weights = np.arange(1, len(x) + 1)
        return float(np.average(x, weights=weights))

    return prices.rolling(window=period, min_periods=1).apply(_weighted_mean, raw=True)


# ---------------------------------------------------------------------------
# Rolling extrema helpers
# ---------------------------------------------------------------------------

def rolling_high(prices: pd.Series, period: int) -> pd.Series:
    """Highest high over *period* observations."""
    return prices.rolling(window=period, min_periods=1).max()


def rolling_low(prices: pd.Series, period: int) -> pd.Series:
    """Lowest low over *period* observations."""
    return prices.rolling(window=period, min_periods=1).min() 