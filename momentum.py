"""
Momentum Indicators - RSI, Stochastic, Williams %R, and momentum oscillators
"""

import pandas as pd
import numpy as np
from typing import Tuple


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI series (0-100)
        
    Example:
        rsi_14 = rsi(hist_data['Close'], 14)
        # RSI > 70 = overbought, RSI < 30 = oversold
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral value


def stochastic(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
               k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        k_period: %K period
        d_period: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    # Calculate %K
    lowest_low = lows.rolling(window=k_period, min_periods=1).min()
    highest_high = highs.rolling(window=k_period, min_periods=1).max()
    
    k_percent = 100 * (closes - lowest_low) / (highest_high - lowest_low)
    
    # Calculate %D (smoothed %K)
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    
    return k_percent.fillna(50), d_percent.fillna(50)


def stochastic_d(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                k_period: int = 14, d_period: int = 3) -> pd.Series:
    """
    Stochastic %D only
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        k_period: %K period
        d_period: %D smoothing period
        
    Returns:
        %D series
    """
    _, d_percent = stochastic(highs, lows, closes, k_period, d_period)
    return d_percent


def williams_r(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Williams %R
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        period: Calculation period
        
    Returns:
        Williams %R series (-100 to 0)
    """
    highest_high = highs.rolling(window=period, min_periods=1).max()
    lowest_low = lows.rolling(window=period, min_periods=1).min()
    
    williams_r = -100 * (highest_high - closes) / (highest_high - lowest_low)
    
    return williams_r.fillna(-50)


def roc(prices: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change (ROC)
    
    Args:
        prices: Price series
        period: ROC period
        
    Returns:
        ROC series as percentage
    """
    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return roc.fillna(0)


def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Price Momentum
    
    Args:
        prices: Price series
        period: Momentum period
        
    Returns:
        Momentum series
    """
    return prices - prices.shift(period)


def cci(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        period: CCI period
        
    Returns:
        CCI series
    """
    # Calculate typical price
    typical_price = (highs + lows + closes) / 3
    
    # Calculate moving average of typical price
    sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
    
    # Calculate mean deviation
    mean_deviation = typical_price.rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    
    # Calculate CCI
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    return cci.fillna(0)


def ultimate_oscillator(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                       period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """
    Ultimate Oscillator
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        period1: Short period
        period2: Medium period
        period3: Long period
        
    Returns:
        Ultimate Oscillator series (0-100)
    """
    # Calculate buying pressure and true range
    prev_close = closes.shift(1)
    buying_pressure = closes - np.minimum(lows, prev_close)
    true_range = np.maximum(highs, prev_close) - np.minimum(lows, prev_close)
    
    # Calculate averages for each period
    bp1 = buying_pressure.rolling(window=period1, min_periods=1).sum()
    tr1 = true_range.rolling(window=period1, min_periods=1).sum()
    
    bp2 = buying_pressure.rolling(window=period2, min_periods=1).sum()
    tr2 = true_range.rolling(window=period2, min_periods=1).sum()
    
    bp3 = buying_pressure.rolling(window=period3, min_periods=1).sum()
    tr3 = true_range.rolling(window=period3, min_periods=1).sum()
    
    # Calculate Ultimate Oscillator
    uo = 100 * ((4 * bp1/tr1) + (2 * bp2/tr2) + (bp3/tr3)) / (4 + 2 + 1)
    
    return uo.fillna(50)


def money_flow_index(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                    volumes: pd.Series, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI) - Volume-weighted RSI
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        volumes: Volume series
        period: MFI period
        
    Returns:
        MFI series (0-100)
    """
    # Calculate typical price and money flow
    typical_price = (highs + lows + closes) / 3
    money_flow = typical_price * volumes
    
    # Calculate positive and negative money flow
    price_change = typical_price.diff()
    positive_mf = money_flow.where(price_change > 0, 0)
    negative_mf = money_flow.where(price_change < 0, 0)
    
    # Calculate money flow ratio
    positive_mf_sum = positive_mf.rolling(window=period, min_periods=1).sum()
    negative_mf_sum = negative_mf.rolling(window=period, min_periods=1).sum()
    
    money_flow_ratio = positive_mf_sum / negative_mf_sum
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi.fillna(50)


def trix(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    TRIX - Triple Exponential Average
    
    Args:
        prices: Price series
        period: Smoothing period
        
    Returns:
        TRIX series
    """
    # Calculate triple smoothed EMA
    ema1 = prices.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # Calculate TRIX as percentage change
    trix = ema3.pct_change() * 100
    
    return trix.fillna(0)


def awesome_oscillator(highs: pd.Series, lows: pd.Series, 
                      fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """
    Awesome Oscillator (AO)
    
    Args:
        highs: High prices series
        lows: Low prices series
        fast_period: Fast SMA period
        slow_period: Slow SMA period
        
    Returns:
        Awesome Oscillator series
    """
    # Calculate median price
    median_price = (highs + lows) / 2
    
    # Calculate SMAs
    fast_sma = median_price.rolling(window=fast_period, min_periods=1).mean()
    slow_sma = median_price.rolling(window=slow_period, min_periods=1).mean()
    
    # Calculate AO
    ao = fast_sma - slow_sma
    
    return ao


def accelerator_oscillator(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                          ao_fast: int = 5, ao_slow: int = 34, signal_period: int = 5) -> pd.Series:
    """
    Accelerator Oscillator (AC)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        ao_fast: AO fast period
        ao_slow: AO slow period
        signal_period: Signal line period
        
    Returns:
        Accelerator Oscillator series
    """
    # Calculate Awesome Oscillator
    ao = awesome_oscillator(highs, lows, ao_fast, ao_slow)
    
    # Calculate signal line (SMA of AO)
    signal_line = ao.rolling(window=signal_period, min_periods=1).mean()
    
    # Calculate AC
    ac = ao - signal_line
    
    return ac 