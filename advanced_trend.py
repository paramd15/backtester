"""
Trend Indicators Module

Advanced trend-following indicators including Parabolic SAR, ADX, and Triple MA systems.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, 
                  af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    Calculate Parabolic SAR (Stop and Reverse)
    
    The Parabolic SAR is a trend-following indicator that provides stop-loss levels
    and potential reversal points for trending markets.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        af_start: Starting acceleration factor (default 0.02)
        af_increment: Acceleration factor increment (default 0.02)
        af_max: Maximum acceleration factor (default 0.2)
        
    Returns:
        Series with Parabolic SAR values
        
    Example:
        >>> psar = parabolic_sar(data['High'], data['Low'])
        >>> # Buy when price crosses above PSAR
        >>> # Sell when price crosses below PSAR
    """
    high = high.copy()
    low = low.copy()
    close = (high + low) / 2  # Use midpoint if close not available
    
    # Initialize arrays
    psar = pd.Series(index=high.index, dtype=float)
    trend = pd.Series(index=high.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    af = pd.Series(index=high.index, dtype=float)
    ep = pd.Series(index=high.index, dtype=float)  # Extreme Point
    
    # Initialize first values
    psar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1
    af.iloc[0] = af_start
    ep.iloc[0] = high.iloc[0]
    
    for i in range(1, len(high)):
        prev_psar = psar.iloc[i-1]
        prev_trend = trend.iloc[i-1]
        prev_af = af.iloc[i-1]
        prev_ep = ep.iloc[i-1]
        
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        
        if prev_trend == 1:  # Previous uptrend
            # Calculate PSAR for uptrend
            current_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            
            # PSAR cannot be above the low of current or previous period
            current_psar = min(current_psar, low.iloc[i-1], current_low)
            
            # Check for trend reversal
            if current_low <= current_psar:
                # Trend reversal to downtrend
                current_trend = -1
                current_psar = prev_ep  # PSAR becomes the previous EP
                current_ep = current_low
                current_af = af_start
            else:
                # Continue uptrend
                current_trend = 1
                if current_high > prev_ep:
                    # New extreme point
                    current_ep = current_high
                    current_af = min(prev_af + af_increment, af_max)
                else:
                    current_ep = prev_ep
                    current_af = prev_af
        
        else:  # Previous downtrend
            # Calculate PSAR for downtrend
            current_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            
            # PSAR cannot be below the high of current or previous period
            current_psar = max(current_psar, high.iloc[i-1], current_high)
            
            # Check for trend reversal
            if current_high >= current_psar:
                # Trend reversal to uptrend
                current_trend = 1
                current_psar = prev_ep  # PSAR becomes the previous EP
                current_ep = current_high
                current_af = af_start
            else:
                # Continue downtrend
                current_trend = -1
                if current_low < prev_ep:
                    # New extreme point
                    current_ep = current_low
                    current_af = min(prev_af + af_increment, af_max)
                else:
                    current_ep = prev_ep
                    current_af = prev_af
        
        psar.iloc[i] = current_psar
        trend.iloc[i] = current_trend
        af.iloc[i] = current_af
        ep.iloc[i] = current_ep
    
    return psar


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """
    Calculate ADX (Average Directional Index) and Directional Movement indicators
    
    ADX measures the strength of a trend, while +DI and -DI indicate direction.
    ADX values above 25 typically indicate a strong trending market.
    
    Args:
        high: Series of high prices
        low: Series of low prices  
        close: Series of closing prices
        period: Period for calculation (default 14)
        
    Returns:
        Tuple of (ADX, +DI, -DI) series
        
    Example:
        >>> adx_val, plus_di, minus_di = adx(data['High'], data['Low'], data['Close'])
        >>> # Strong trend when ADX > 25
        >>> # Uptrend when +DI > -DI and ADX > 25
        >>> # Downtrend when -DI > +DI and ADX > 25
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    # Only keep positive movements, others set to 0
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Calculate smoothed TR and DM using Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # Calculate DX (Directional Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX (Average Directional Index)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx_val, plus_di, minus_di


def triple_moving_average(close: pd.Series, ma1_period: int = 5, ma2_period: int = 13, 
                         ma3_period: int = 21, ma_type: str = 'sma') -> tuple:
    """
    Calculate Triple Moving Average system
    
    A trend-following system using three moving averages of different periods.
    Signals are generated when all three MAs are aligned in the same direction.
    
    Args:
        close: Series of closing prices
        ma1_period: Period for fastest MA (default 5)
        ma2_period: Period for medium MA (default 13)
        ma3_period: Period for slowest MA (default 21)
        ma_type: Type of MA ('sma', 'ema', 'wma') (default 'sma')
        
    Returns:
        Tuple of (MA1, MA2, MA3, Signal) where Signal is 1 for bullish, -1 for bearish, 0 for neutral
        
    Example:
        >>> ma1, ma2, ma3, signal = triple_moving_average(data['Close'])
        >>> # Buy when signal == 1 (all MAs bullishly aligned)
        >>> # Sell when signal == -1 (all MAs bearishly aligned)
    """
    if ma_type.lower() == 'sma':
        ma1 = close.rolling(window=ma1_period).mean()
        ma2 = close.rolling(window=ma2_period).mean()
        ma3 = close.rolling(window=ma3_period).mean()
    elif ma_type.lower() == 'ema':
        ma1 = close.ewm(span=ma1_period).mean()
        ma2 = close.ewm(span=ma2_period).mean()
        ma3 = close.ewm(span=ma3_period).mean()
    elif ma_type.lower() == 'wma':
        ma1 = close.rolling(window=ma1_period).apply(lambda x: np.average(x, weights=range(1, len(x)+1)))
        ma2 = close.rolling(window=ma2_period).apply(lambda x: np.average(x, weights=range(1, len(x)+1)))
        ma3 = close.rolling(window=ma3_period).apply(lambda x: np.average(x, weights=range(1, len(x)+1)))
    else:
        raise ValueError("ma_type must be 'sma', 'ema', or 'wma'")
    
    # Generate signals based on MA alignment
    signal = pd.Series(0, index=close.index)
    
    # Bullish alignment: MA1 > MA2 > MA3 (fast above medium above slow)
    bullish = (ma1 > ma2) & (ma2 > ma3)
    
    # Bearish alignment: MA1 < MA2 < MA3 (fast below medium below slow)
    bearish = (ma1 < ma2) & (ma2 < ma3)
    
    signal[bullish] = 1
    signal[bearish] = -1
    
    return ma1, ma2, ma3, signal


# Additional trend strength indicators

def trend_strength_index(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Trend Strength Index
    
    Measures the strength of the current trend by comparing the current price
    to recent high/low ranges. Values near 1 indicate strong uptrend,
    values near 0 indicate strong downtrend.
    
    Args:
        close: Series of closing prices
        period: Lookback period (default 20)
        
    Returns:
        Series with trend strength values (0 to 1)
    """
    high_period = close.rolling(window=period).max()
    low_period = close.rolling(window=period).min()
    
    # Calculate trend strength as position within the range
    trend_strength = (close - low_period) / (high_period - low_period)
    
    return trend_strength


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
               period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Calculate Supertrend indicator
    
    Supertrend is a trend-following indicator that uses ATR to set dynamic
    support and resistance levels. It provides clear buy/sell signals.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)
        
    Returns:
        Tuple of (Supertrend, Signal) where Signal is 1 for buy, -1 for sell
        
    Example:
        >>> st, signal = supertrend(data['High'], data['Low'], data['Close'])
        >>> # Buy when signal changes from -1 to 1
        >>> # Sell when signal changes from 1 to -1
    """
    # Calculate ATR
    hl2 = (high + low) / 2
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate basic upper and lower bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize final bands
    final_upper_band = pd.Series(index=close.index, dtype=float)
    final_lower_band = pd.Series(index=close.index, dtype=float)
    supertrend = pd.Series(index=close.index, dtype=float)
    signal = pd.Series(index=close.index, dtype=int)
    
    # Set initial values
    final_upper_band.iloc[0] = upper_band.iloc[0]
    final_lower_band.iloc[0] = lower_band.iloc[0]
    supertrend.iloc[0] = lower_band.iloc[0]
    signal.iloc[0] = 1
    
    for i in range(1, len(close)):
        # Final upper band
        if upper_band.iloc[i] < final_upper_band.iloc[i-1] or close.iloc[i-1] > final_upper_band.iloc[i-1]:
            final_upper_band.iloc[i] = upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
        
        # Final lower band
        if lower_band.iloc[i] > final_lower_band.iloc[i-1] or close.iloc[i-1] < final_lower_band.iloc[i-1]:
            final_lower_band.iloc[i] = lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        
        # Supertrend calculation
        if supertrend.iloc[i-1] == final_upper_band.iloc[i-1] and close.iloc[i] <= final_upper_band.iloc[i]:
            supertrend.iloc[i] = final_upper_band.iloc[i]
        elif supertrend.iloc[i-1] == final_upper_band.iloc[i-1] and close.iloc[i] > final_upper_band.iloc[i]:
            supertrend.iloc[i] = final_lower_band.iloc[i]
        elif supertrend.iloc[i-1] == final_lower_band.iloc[i-1] and close.iloc[i] >= final_lower_band.iloc[i]:
            supertrend.iloc[i] = final_lower_band.iloc[i]
        elif supertrend.iloc[i-1] == final_lower_band.iloc[i-1] and close.iloc[i] < final_lower_band.iloc[i]:
            supertrend.iloc[i] = final_upper_band.iloc[i]
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
        
        # Signal generation
        if close.iloc[i] > supertrend.iloc[i]:
            signal.iloc[i] = 1  # Buy signal
        else:
            signal.iloc[i] = -1  # Sell signal
    
    return supertrend, signal 