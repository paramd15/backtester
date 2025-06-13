"""
Trend Indicators - Moving averages, MACD, Bollinger Bands, and trend analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union
from .common import sma as _sma, ema as _ema, wma as _wma


def sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average
    
    Args:
        prices: Price series
        period: Number of periods
        
    Returns:
        SMA series
    """
    return _sma(prices, period)


def ema(prices: pd.Series, period: int, alpha: float = None) -> pd.Series:
    """
    Exponential Moving Average
    
    Args:
        prices: Price series
        period: Number of periods
        alpha: Smoothing factor (if None, calculated as 2/(period+1))
        
    Returns:
        EMA series
    """
    if alpha is None:
        # Let the common helper calculate the default alpha
        return _ema(prices, period)
    else:
        return _ema(prices, period, alpha=alpha)


def wma(prices: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average (linear weights)
    
    Args:
        prices: Price series
        period: Number of periods
        
    Returns:
        WMA series
    """
    return _wma(prices, period)


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def macd_signal(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    MACD Signal Line only
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        MACD signal line
    """
    _, signal_line, _ = macd(prices, fast, slow, signal)
    return signal_line


def macd_histogram(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    MACD Histogram only
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        MACD histogram
    """
    _, _, histogram = macd(prices, fast, slow, signal)
    return histogram


def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands
    
    Args:
        prices: Price series
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (Upper band, Middle band/SMA, Lower band)
    """
    middle_band = sma(prices, period)
    std = prices.rolling(window=period, min_periods=1).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def bollinger_width(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Bollinger Band Width (volatility measure)
    
    Args:
        prices: Price series
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        Band width series
    """
    upper_band, middle_band, lower_band = bollinger_bands(prices, period, std_dev)
    return (upper_band - lower_band) / middle_band


def moving_average_envelope(prices: pd.Series, period: int = 20, envelope_pct: float = 0.025) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Envelope
    
    Args:
        prices: Price series
        period: Moving average period
        envelope_pct: Envelope percentage (0.025 = 2.5%)
        
    Returns:
        Tuple of (Upper envelope, Middle/SMA, Lower envelope)
    """
    middle = sma(prices, period)
    envelope = middle * envelope_pct
    
    upper_envelope = middle + envelope
    lower_envelope = middle - envelope
    
    return upper_envelope, middle, lower_envelope


def trend_strength(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Trend Strength Indicator
    
    Args:
        prices: Price series
        period: Calculation period
        
    Returns:
        Trend strength series (0-100)
    """
    # Calculate price changes
    changes = prices.diff()
    
    # Separate positive and negative changes
    gains = changes.where(changes > 0, 0)
    losses = -changes.where(changes < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    
    # Calculate trend strength
    trend_strength = 100 * avg_gains / (avg_gains + avg_losses)
    
    return trend_strength.fillna(50)  # Fill NaN with neutral value


def price_channels(highs: pd.Series, lows: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Price Channels (Highest High and Lowest Low)
    
    Args:
        highs: High prices series
        lows: Low prices series
        period: Lookback period
        
    Returns:
        Tuple of (Upper channel, Lower channel)
    """
    upper_channel = highs.rolling(window=period, min_periods=1).max()
    lower_channel = lows.rolling(window=period, min_periods=1).min()
    
    return upper_channel, lower_channel


def linear_regression_slope(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Linear Regression Slope (trend direction)
    
    Args:
        prices: Price series
        period: Regression period
        
    Returns:
        Slope series (positive = uptrend, negative = downtrend)
    """
    def calculate_slope(y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    return prices.rolling(window=period, min_periods=2).apply(calculate_slope, raw=True)


def trend_direction(prices: pd.Series, short_period: int = 10, long_period: int = 30) -> pd.Series:
    """
    Trend Direction Indicator
    
    Args:
        prices: Price series
        short_period: Short-term MA period
        long_period: Long-term MA period
        
    Returns:
        Trend direction series (1 = uptrend, -1 = downtrend, 0 = sideways)
    """
    short_ma = sma(prices, short_period)
    long_ma = sma(prices, long_period)
    
    trend = pd.Series(index=prices.index, dtype=float)
    trend[short_ma > long_ma] = 1   # Uptrend
    trend[short_ma < long_ma] = -1  # Downtrend
    trend[short_ma == long_ma] = 0  # Sideways
    
    return trend.fillna(0) 