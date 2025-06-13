"""
Volume Indicators - Volume analysis and volume-based indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple


def volume_sma(volumes: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Simple Moving Average
    
    Args:
        volumes: Volume series
        period: SMA period
        
    Returns:
        Volume SMA series
    """
    return volumes.rolling(window=period, min_periods=1).mean()


def volume_ratio(volumes: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Ratio (Current Volume / Average Volume)
    
    Args:
        volumes: Volume series
        period: Average volume period
        
    Returns:
        Volume ratio series (>1 = above average volume)
    """
    avg_volume = volume_sma(volumes, period)
    return volumes / avg_volume


def on_balance_volume(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV)
    
    Args:
        closes: Close prices series
        volumes: Volume series
        
    Returns:
        OBV series
        
    Example:
        obv = on_balance_volume(hist_data['Close'], hist_data['Volume'])
        # Rising OBV with rising prices = bullish
        # Falling OBV with rising prices = bearish divergence
    """
    # Calculate price changes
    price_change = closes.diff()
    
    # Determine volume direction
    volume_direction = pd.Series(index=closes.index, dtype=float)
    volume_direction[price_change > 0] = volumes[price_change > 0]    # Up volume
    volume_direction[price_change < 0] = -volumes[price_change < 0]   # Down volume
    volume_direction[price_change == 0] = 0                          # Unchanged volume
    
    # Calculate cumulative OBV
    obv = volume_direction.cumsum()
    
    return obv


def accumulation_distribution(highs: pd.Series, lows: pd.Series, closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Accumulation/Distribution Line (A/D Line)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        volumes: Volume series
        
    Returns:
        A/D Line series
    """
    # Calculate Money Flow Multiplier
    clv = ((closes - lows) - (highs - closes)) / (highs - lows)
    clv = clv.fillna(0)  # Handle division by zero when high == low
    
    # Calculate Money Flow Volume
    money_flow_volume = clv * volumes
    
    # Calculate A/D Line (cumulative)
    ad_line = money_flow_volume.cumsum()
    
    return ad_line


def volume_price_trend(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Volume Price Trend (VPT)
    
    Args:
        closes: Close prices series
        volumes: Volume series
        
    Returns:
        VPT series
    """
    # Calculate price change percentage
    price_change_pct = closes.pct_change()
    
    # Calculate VPT
    vpt = (price_change_pct * volumes).cumsum()
    
    return vpt.fillna(0)


def chaikin_money_flow(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                      volumes: pd.Series, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        volumes: Volume series
        period: CMF period
        
    Returns:
        CMF series (-1 to +1)
    """
    # Calculate Money Flow Multiplier
    clv = ((closes - lows) - (highs - closes)) / (highs - lows)
    clv = clv.fillna(0)
    
    # Calculate Money Flow Volume
    money_flow_volume = clv * volumes
    
    # Calculate CMF
    cmf = (money_flow_volume.rolling(window=period, min_periods=1).sum() / 
           volumes.rolling(window=period, min_periods=1).sum())
    
    return cmf


def volume_oscillator(volumes: pd.Series, short_period: int = 5, long_period: int = 10) -> pd.Series:
    """
    Volume Oscillator
    
    Args:
        volumes: Volume series
        short_period: Short MA period
        long_period: Long MA period
        
    Returns:
        Volume oscillator series
    """
    short_ma = volume_sma(volumes, short_period)
    long_ma = volume_sma(volumes, long_period)
    
    volume_osc = ((short_ma - long_ma) / long_ma) * 100
    
    return volume_osc


def ease_of_movement(highs: pd.Series, lows: pd.Series, volumes: pd.Series, period: int = 14) -> pd.Series:
    """
    Ease of Movement (EOM)
    
    Args:
        highs: High prices series
        lows: Low prices series
        volumes: Volume series
        period: Smoothing period
        
    Returns:
        EOM series
    """
    # Calculate distance moved
    distance_moved = ((highs + lows) / 2) - ((highs.shift(1) + lows.shift(1)) / 2)
    
    # Calculate box height
    box_height = (volumes / 100000000) / (highs - lows)  # Scale volume
    
    # Calculate 1-period EMV
    emv = distance_moved / box_height
    emv = emv.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate smoothed EOM
    eom = emv.rolling(window=period, min_periods=1).mean()
    
    return eom


def negative_volume_index(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Negative Volume Index (NVI)
    
    Args:
        closes: Close prices series
        volumes: Volume series
        
    Returns:
        NVI series
    """
    nvi = pd.Series(index=closes.index, dtype=float)
    nvi.iloc[0] = 1000  # Starting value
    
    for i in range(1, len(closes)):
        if volumes.iloc[i] < volumes.iloc[i-1]:  # Volume decreased
            nvi.iloc[i] = nvi.iloc[i-1] * (closes.iloc[i] / closes.iloc[i-1])
        else:  # Volume same or increased
            nvi.iloc[i] = nvi.iloc[i-1]
    
    return nvi


def positive_volume_index(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Positive Volume Index (PVI)
    
    Args:
        closes: Close prices series
        volumes: Volume series
        
    Returns:
        PVI series
    """
    pvi = pd.Series(index=closes.index, dtype=float)
    pvi.iloc[0] = 1000  # Starting value
    
    for i in range(1, len(closes)):
        if volumes.iloc[i] > volumes.iloc[i-1]:  # Volume increased
            pvi.iloc[i] = pvi.iloc[i-1] * (closes.iloc[i] / closes.iloc[i-1])
        else:  # Volume same or decreased
            pvi.iloc[i] = pvi.iloc[i-1]
    
    return pvi


def volume_weighted_average_price(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                                 volumes: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP)
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        volumes: Volume series
        period: VWAP period
        
    Returns:
        VWAP series
    """
    # Calculate typical price
    typical_price = (highs + lows + closes) / 3
    
    # Calculate price * volume
    pv = typical_price * volumes
    
    # Calculate VWAP
    vwap = (pv.rolling(window=period, min_periods=1).sum() / 
            volumes.rolling(window=period, min_periods=1).sum())
    
    return vwap


def volume_rate_of_change(volumes: pd.Series, period: int = 12) -> pd.Series:
    """
    Volume Rate of Change (VROC)
    
    Args:
        volumes: Volume series
        period: ROC period
        
    Returns:
        VROC series as percentage
    """
    vroc = ((volumes - volumes.shift(period)) / volumes.shift(period)) * 100
    return vroc.fillna(0)


def klinger_oscillator(highs: pd.Series, lows: pd.Series, closes: pd.Series, volumes: pd.Series,
                      fast_period: int = 34, slow_period: int = 55, signal_period: int = 13) -> Tuple[pd.Series, pd.Series]:
    """
    Klinger Oscillator
    
    Args:
        highs: High prices series
        lows: Low prices series
        closes: Close prices series
        volumes: Volume series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (Klinger Oscillator, Signal Line)
    """
    # Calculate typical price
    hlc = (highs + lows + closes) / 3
    
    # Calculate trend
    trend = pd.Series(index=closes.index, dtype=int)
    trend[hlc > hlc.shift(1)] = 1
    trend[hlc < hlc.shift(1)] = -1
    trend[hlc == hlc.shift(1)] = trend.shift(1)
    trend = trend.fillna(1)
    
    # Calculate volume force
    volume_force = volumes * trend * abs(2 * ((closes - lows) - (highs - closes)) / (highs - lows) - 1)
    volume_force = volume_force.fillna(0)
    
    # Calculate Klinger Oscillator
    fast_ema = volume_force.ewm(span=fast_period, adjust=False).mean()
    slow_ema = volume_force.ewm(span=slow_period, adjust=False).mean()
    
    klinger = fast_ema - slow_ema
    signal_line = klinger.ewm(span=signal_period, adjust=False).mean()
    
    return klinger, signal_line


def volume_profile_analysis(closes: pd.Series, volumes: pd.Series, price_bins: int = 50) -> pd.DataFrame:
    """
    Volume Profile Analysis
    
    Args:
        closes: Close prices series
        volumes: Volume series
        price_bins: Number of price bins
        
    Returns:
        DataFrame with price levels and corresponding volumes
    """
    # Create price bins
    price_min = closes.min()
    price_max = closes.max()
    price_levels = np.linspace(price_min, price_max, price_bins)
    
    # Assign each price to a bin
    price_bins_assigned = pd.cut(closes, bins=price_levels, include_lowest=True)
    
    # Calculate volume for each price level
    volume_profile = volumes.groupby(price_bins_assigned).sum().sort_index()
    
    # Create result DataFrame
    result = pd.DataFrame({
        'price_level': [interval.mid for interval in volume_profile.index],
        'volume': volume_profile.values,
        'volume_pct': (volume_profile.values / volume_profile.sum()) * 100
    })
    
    return result.sort_values('volume', ascending=False) 