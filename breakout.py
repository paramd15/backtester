"""
Breakout and Support/Resistance Indicators
Perfect for breakout strategies like "buy on 20-day high breakout"
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from .common import rolling_high as _rolling_high, rolling_low as _rolling_low


def rolling_high(prices: pd.Series, period: int) -> pd.Series:
    """
    Rolling Maximum (Highest High over N periods)
    Perfect for breakout strategies
    
    Args:
        prices: Price series (typically High prices)
        period: Number of periods to look back
        
    Returns:
        Rolling high series
        
    Example:
        # 20-day high for breakout strategy
        twenty_day_high = rolling_high(hist_data['High'], 20)
    """
    return _rolling_high(prices, period)


def rolling_low(prices: pd.Series, period: int) -> pd.Series:
    """
    Rolling Minimum (Lowest Low over N periods)
    Perfect for breakdown/exit strategies
    
    Args:
        prices: Price series (typically Low prices)
        period: Number of periods to look back
        
    Returns:
        Rolling low series
        
    Example:
        # 10-day low for exit strategy
        ten_day_low = rolling_low(hist_data['Low'], 10)
    """
    return _rolling_low(prices, period)


def donchian_channels(highs: pd.Series, lows: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels (Turtle Trading System)
    
    Args:
        highs: High prices series
        lows: Low prices series
        period: Channel period
        
    Returns:
        Tuple of (Upper channel, Middle channel, Lower channel)
        
    Example:
        upper, middle, lower = donchian_channels(hist_data['High'], hist_data['Low'], 20)
        # Buy on breakout above upper channel
        # Sell on breakdown below lower channel
    """
    upper_channel = rolling_high(highs, period)
    lower_channel = rolling_low(lows, period)
    middle_channel = (upper_channel + lower_channel) / 2
    
    return upper_channel, middle_channel, lower_channel


def donchian_width(highs: pd.Series, lows: pd.Series, period: int = 20) -> pd.Series:
    """
    Donchian Channel Width (volatility measure)
    
    Args:
        highs: High prices series
        lows: Low prices series
        period: Channel period
        
    Returns:
        Channel width series
    """
    upper_channel = rolling_high(highs, period)
    lower_channel = rolling_low(lows, period)
    
    return upper_channel - lower_channel


def breakout_strength(prices: pd.Series, highs: pd.Series, period: int = 20) -> pd.Series:
    """
    Breakout Strength - How far price is above the breakout level
    
    Args:
        prices: Current prices (typically Close)
        highs: High prices series
        period: Breakout period
        
    Returns:
        Breakout strength as percentage above breakout level
        
    Example:
        strength = breakout_strength(hist_data['Close'], hist_data['High'], 20)
        # Values > 0 indicate breakout above 20-day high
    """
    breakout_level = rolling_high(highs, period).shift(1)  # Previous period's high
    return (prices - breakout_level) / breakout_level


def support_resistance_levels(prices: pd.Series, window: int = 20, min_touches: int = 2) -> Tuple[List[float], List[float]]:
    """
    Identify Support and Resistance Levels
    
    Args:
        prices: Price series
        window: Window for local extrema detection
        min_touches: Minimum touches to confirm level
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    # Find local maxima and minima
    highs = prices.rolling(window=window, center=True).max()
    lows = prices.rolling(window=window, center=True).min()
    
    # Identify peaks and troughs
    peaks = prices[(prices == highs) & (prices.shift(1) < prices) & (prices.shift(-1) < prices)]
    troughs = prices[(prices == lows) & (prices.shift(1) > prices) & (prices.shift(-1) > prices)]
    
    # Group similar levels
    def group_levels(levels, tolerance=0.02):
        if len(levels) == 0:
            return []
        
        grouped = []
        levels_sorted = sorted(levels)
        current_group = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                current_group.append(level)
            else:
                if len(current_group) >= min_touches:
                    grouped.append(np.mean(current_group))
                current_group = [level]
        
        if len(current_group) >= min_touches:
            grouped.append(np.mean(current_group))
        
        return grouped
    
    resistance_levels = group_levels(peaks.tolist())
    support_levels = group_levels(troughs.tolist())
    
    return support_levels, resistance_levels


def pivot_points(highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Classic Pivot Points (Daily)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        
    Returns:
        Tuple of (Pivot, R1, R2, S1, S2)
    """
    # Use previous day's data
    prev_high = highs.shift(1)
    prev_low = lows.shift(1)
    prev_close = closes.shift(1)
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    
    # Calculate resistance and support levels
    r1 = 2 * pivot - prev_low
    r2 = pivot + (prev_high - prev_low)
    s1 = 2 * pivot - prev_high
    s2 = pivot - (prev_high - prev_low)
    
    return pivot, r1, r2, s1, s2


def fibonacci_retracement(high_price: float, low_price: float) -> dict:
    """
    Calculate Fibonacci Retracement Levels
    
    Args:
        high_price: Swing high price
        low_price: Swing low price
        
    Returns:
        Dictionary of Fibonacci levels
    """
    diff = high_price - low_price
    
    levels = {
        '0%': high_price,
        '23.6%': high_price - 0.236 * diff,
        '38.2%': high_price - 0.382 * diff,
        '50%': high_price - 0.5 * diff,
        '61.8%': high_price - 0.618 * diff,
        '78.6%': high_price - 0.786 * diff,
        '100%': low_price
    }
    
    return levels


def breakout_confirmation(prices: pd.Series, volumes: pd.Series, breakout_level: pd.Series, 
                         volume_threshold: float = 1.5) -> pd.Series:
    """
    Confirm breakouts with volume
    
    Args:
        prices: Price series
        volumes: Volume series
        breakout_level: Breakout level series
        volume_threshold: Volume multiplier for confirmation
        
    Returns:
        Boolean series indicating confirmed breakouts
    """
    # Calculate average volume
    avg_volume = volumes.rolling(window=20, min_periods=1).mean()
    
    # Price breakout condition
    price_breakout = prices > breakout_level
    
    # Volume confirmation
    volume_confirmation = volumes > (avg_volume * volume_threshold)
    
    # Confirmed breakout
    confirmed_breakout = price_breakout & volume_confirmation
    
    return confirmed_breakout


def gap_detection(opens: pd.Series, closes: pd.Series, threshold: float = 0.02) -> pd.Series:
    """
    Detect price gaps
    
    Args:
        opens: Open prices series
        closes: Close prices series
        threshold: Minimum gap size as percentage
        
    Returns:
        Gap size series (positive for gap up, negative for gap down)
    """
    prev_close = closes.shift(1)
    gap_size = (opens - prev_close) / prev_close
    
    # Only return significant gaps
    significant_gaps = gap_size.where(abs(gap_size) >= threshold, 0)
    
    return significant_gaps


def consolidation_detection(highs: pd.Series, lows: pd.Series, period: int = 10, 
                          max_range: float = 0.05) -> pd.Series:
    """
    Detect consolidation periods (sideways movement)
    
    Args:
        highs: High prices series
        lows: Low prices series
        period: Period to check for consolidation
        max_range: Maximum range as percentage for consolidation
        
    Returns:
        Boolean series indicating consolidation periods
    """
    # Calculate range over period
    period_high = rolling_high(highs, period)
    period_low = rolling_low(lows, period)
    price_range = (period_high - period_low) / period_low
    
    # Consolidation when range is small
    consolidation = price_range <= max_range
    
    return consolidation


def trend_line_break(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Detect trend line breaks using linear regression
    
    Args:
        prices: Price series
        period: Period for trend line calculation
        
    Returns:
        Trend line break signals (1 = upward break, -1 = downward break, 0 = no break)
    """
    def calculate_trend_line(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(y) - 1) + intercept  # Trend line value at current point
    
    # Calculate trend line
    trend_line = prices.rolling(window=period, min_periods=2).apply(calculate_trend_line, raw=True)
    
    # Detect breaks
    breaks = pd.Series(index=prices.index, dtype=float)
    breaks[(prices > trend_line) & (prices.shift(1) <= trend_line.shift(1))] = 1   # Upward break
    breaks[(prices < trend_line) & (prices.shift(1) >= trend_line.shift(1))] = -1  # Downward break
    breaks = breaks.fillna(0)
    
    return breaks 