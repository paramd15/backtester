"""
Volatility Indicators - ATR, volatility measures, and volatility-based channels
"""

import pandas as pd
import numpy as np
from typing import Tuple


def true_range(highs: pd.Series, lows: pd.Series, closes: pd.Series) -> pd.Series:
    """
    True Range calculation
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        
    Returns:
        True Range series
    """
    prev_close = closes.shift(1)
    
    # Calculate the three components of True Range
    tr1 = highs - lows
    tr2 = abs(highs - prev_close)
    tr3 = abs(lows - prev_close)
    
    # True Range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return true_range


def atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        period: ATR period
        
    Returns:
        ATR series
        
    Example:
        atr_14 = atr(hist_data['High'], hist_data['Low'], hist_data['Close'], 14)
        # Higher ATR = higher volatility
    """
    tr = true_range(highs, lows, closes)
    
    # Use Wilder's smoothing (similar to EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Historical Volatility (Standard Deviation of Returns)
    
    Args:
        prices: Price series
        period: Calculation period
        
    Returns:
        Volatility series
    """
    returns = prices.pct_change()
    volatility = returns.rolling(window=period, min_periods=1).std()
    
    return volatility


def historical_volatility(prices: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """
    Historical Volatility (annualized)
    
    Args:
        prices: Price series
        period: Calculation period
        annualize: Whether to annualize the volatility
        
    Returns:
        Historical volatility series
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=period, min_periods=1).std()
    
    if annualize:
        vol = vol * np.sqrt(252)  # Annualize assuming 252 trading days
    
    return vol


def keltner_channels(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                    period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        period: EMA and ATR period
        multiplier: ATR multiplier
        
    Returns:
        Tuple of (Upper channel, Middle line/EMA, Lower channel)
    """
    # Calculate middle line (EMA of close)
    middle_line = closes.ewm(span=period, adjust=False).mean()
    
    # Calculate ATR
    atr_values = atr(highs, lows, closes, period)
    
    # Calculate channels
    upper_channel = middle_line + (multiplier * atr_values)
    lower_channel = middle_line - (multiplier * atr_values)
    
    return upper_channel, middle_line, lower_channel


def average_true_range_percent(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                              period: int = 14) -> pd.Series:
    """
    ATR as percentage of price
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        period: ATR period
        
    Returns:
        ATR percentage series
    """
    atr_values = atr(highs, lows, closes, period)
    atr_percent = (atr_values / closes) * 100
    
    return atr_percent


def volatility_ratio(prices: pd.Series, short_period: int = 10, long_period: int = 30) -> pd.Series:
    """
    Volatility Ratio (short-term vs long-term volatility)
    
    Args:
        prices: Price series
        short_period: Short-term volatility period
        long_period: Long-term volatility period
        
    Returns:
        Volatility ratio series (>1 = increasing volatility, <1 = decreasing volatility)
    """
    short_vol = volatility(prices, short_period)
    long_vol = volatility(prices, long_period)
    
    vol_ratio = short_vol / long_vol
    
    return vol_ratio


def volatility_breakout(prices: pd.Series, period: int = 20, threshold: float = 2.0) -> pd.Series:
    """
    Volatility Breakout Indicator
    
    Args:
        prices: Price series
        period: Volatility calculation period
        threshold: Volatility threshold multiplier
        
    Returns:
        Boolean series indicating volatility breakouts
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=period, min_periods=1).std()
    avg_vol = vol.rolling(window=period, min_periods=1).mean()
    
    # Volatility breakout when current volatility exceeds threshold
    breakout = vol > (avg_vol * threshold)
    
    return breakout


def chaikin_volatility(highs: pd.Series, lows: pd.Series, period: int = 10, roc_period: int = 10) -> pd.Series:
    """
    Chaikin Volatility
    
    Args:
        highs: High prices series
        lows: Low prices series
        period: EMA period for H-L
        roc_period: Rate of change period
        
    Returns:
        Chaikin Volatility series
    """
    # Calculate H-L spread
    hl_spread = highs - lows
    
    # Calculate EMA of H-L spread
    ema_hl = hl_spread.ewm(span=period, adjust=False).mean()
    
    # Calculate rate of change of EMA
    chaikin_vol = ((ema_hl - ema_hl.shift(roc_period)) / ema_hl.shift(roc_period)) * 100
    
    return chaikin_vol.fillna(0)


def volatility_system_entry(prices: pd.Series, highs: pd.Series, lows: pd.Series, closes: pd.Series,
                           atr_period: int = 20, atr_multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """
    Volatility-based Entry System (Turtle-style)
    
    Args:
        prices: Price series (typically Close)
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        atr_period: ATR calculation period
        atr_multiplier: ATR multiplier for stops
        
    Returns:
        Tuple of (Entry signals, Stop levels)
    """
    # Calculate ATR
    atr_values = atr(highs, lows, closes, atr_period)
    
    # Calculate 20-day breakout levels
    breakout_high = highs.rolling(window=20, min_periods=1).max().shift(1)
    breakout_low = lows.rolling(window=20, min_periods=1).min().shift(1)
    
    # Entry signals
    long_entry = prices > breakout_high
    short_entry = prices < breakout_low
    
    # Combine entry signals
    entry_signals = pd.Series(index=prices.index, dtype=float)
    entry_signals[long_entry] = 1   # Long entry
    entry_signals[short_entry] = -1  # Short entry
    entry_signals = entry_signals.fillna(0)
    
    # Calculate stop levels
    long_stop = prices - (atr_multiplier * atr_values)
    short_stop = prices + (atr_multiplier * atr_values)
    
    # Set stop levels based on position
    stop_levels = pd.Series(index=prices.index, dtype=float)
    stop_levels[entry_signals == 1] = long_stop[entry_signals == 1]
    stop_levels[entry_signals == -1] = short_stop[entry_signals == -1]
    
    return entry_signals, stop_levels


def volatility_adjusted_momentum(prices: pd.Series, highs: pd.Series, lows: pd.Series, closes: pd.Series,
                                momentum_period: int = 12, atr_period: int = 14) -> pd.Series:
    """
    Volatility-Adjusted Momentum
    
    Args:
        prices: Price series
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        momentum_period: Momentum calculation period
        atr_period: ATR period for volatility adjustment
        
    Returns:
        Volatility-adjusted momentum series
    """
    # Calculate raw momentum
    raw_momentum = prices - prices.shift(momentum_period)
    
    # Calculate ATR for volatility adjustment
    atr_values = atr(highs, lows, closes, atr_period)
    
    # Adjust momentum by volatility
    vol_adj_momentum = raw_momentum / atr_values
    
    return vol_adj_momentum


def volatility_cone(prices: pd.Series, periods: list = [10, 20, 30, 60, 90]) -> pd.DataFrame:
    """
    Volatility Cone Analysis
    
    Args:
        prices: Price series
        periods: List of periods to calculate volatility for
        
    Returns:
        DataFrame with volatility statistics for each period
    """
    results = {}
    
    for period in periods:
        vol = historical_volatility(prices, period, annualize=True)
        
        results[f'{period}d'] = {
            'current': vol.iloc[-1] if len(vol) > 0 else np.nan,
            'mean': vol.mean(),
            'std': vol.std(),
            'min': vol.min(),
            'max': vol.max(),
            'percentile_10': vol.quantile(0.1),
            'percentile_25': vol.quantile(0.25),
            'percentile_75': vol.quantile(0.75),
            'percentile_90': vol.quantile(0.9)
        }
    
    return pd.DataFrame(results).T 