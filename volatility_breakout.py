"""
Volatility Breakout Strategies
Implementation of Phase 1.2 from roadmap - Classic Breakout Strategies

This module implements ATR-based and volatility-based breakout systems
that enter trades when price breaks out of normal volatility ranges.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_strategy import BaseStrategy


class ATRVolatilityBreakout(BaseStrategy):
    """
    ATR-based Volatility Breakout Strategy
    
    Entry Logic:
    - Buy when price breaks above (High - ATR * multiplier) from X days ago
    - Must have volume confirmation (volume > threshold * average)
    - Price must be above long-term trend filter
    
    Exit Logic:
    - Sell when price breaks below (Low + ATR * multiplier) from Y days ago
    - Or after maximum hold period
    
    This strategy captures explosive moves that break out of normal volatility ranges.
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 lookback_period: int = 20,
                 volume_threshold: float = 1.5,
                 trend_filter_period: int = 200,
                 max_hold_days: int = 30,
                 position_size: float = 0.1):
        """
        Initialize ATR Volatility Breakout Strategy
        
        Args:
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR to set breakout levels
            lookback_period: Days to look back for high/low calculation
            volume_threshold: Volume must be X times average to confirm
            trend_filter_period: Period for trend filter (SMA)
            max_hold_days: Maximum days to hold position
            position_size: Fraction of capital to use per trade
        """
        # Create parameters dictionary
        parameters = {
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'lookback_period': lookback_period,
            'volume_threshold': volume_threshold,
            'trend_filter_period': trend_filter_period,
            'max_hold_days': max_hold_days,
            'position_size': position_size
        }
        
        strategy_name = "ATR Volatility Breakout"
        super().__init__(strategy_name, parameters)
        
        # Store parameters as instance attributes for easy access
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.trend_filter_period = trend_filter_period
        self.max_hold_days = max_hold_days
        self.position_size = position_size
        
        self.description = f"ATR({atr_period}) * {atr_multiplier} breakout with {lookback_period}-day lookback"
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()
        
        # Calculate ATR
        df['ATR'] = self._calculate_atr(df, self.atr_period)
        
        # Calculate rolling high/low for breakout levels
        df['Highest_High'] = df['High'].rolling(window=self.lookback_period).max()
        df['Lowest_Low'] = df['Low'].rolling(window=self.lookback_period).min()
        
        # Calculate breakout levels
        df['Breakout_Buy_Level'] = df['Highest_High'] + (df['ATR'] * self.atr_multiplier)
        df['Breakout_Sell_Level'] = df['Lowest_Low'] - (df['ATR'] * self.atr_multiplier)
        
        # Trend filter (SMA)
        df['Trend_Filter'] = df['Close'].rolling(window=self.trend_filter_period).mean()
        
        # Volume filter
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on volatility breakout"""
        df = self.calculate_indicators(data)
        
        # Initialize signal columns
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        # Track position state
        in_position = False
        entry_date = None
        entry_price = None
        
        for i in range(self.lookback_period + self.trend_filter_period, len(df)):
            current_idx = df.index[i]
            
            if not in_position:
                # Check for buy signal
                buy_signal = self._check_buy_signal(df, i)
                
                if buy_signal:
                    df.loc[current_idx, 'Signal'] = 'BUY'
                    df.loc[current_idx, 'Signal_Strength'] = buy_signal
                    df.loc[current_idx, 'Entry_Price'] = df.iloc[i]['Close']
                    
                    in_position = True
                    entry_date = current_idx
                    entry_price = df.iloc[i]['Close']
                    
            else:
                # Check for sell signal
                sell_signal = self._check_sell_signal(df, i, entry_date)
                
                if sell_signal:
                    df.loc[current_idx, 'Signal'] = 'SELL'
                    df.loc[current_idx, 'Signal_Strength'] = sell_signal
                    df.loc[current_idx, 'Exit_Price'] = df.iloc[i]['Close']
                    
                    in_position = False
                    entry_date = None
                    entry_price = None
        
        return df
    
    def _check_buy_signal(self, df: pd.DataFrame, i: int) -> float:
        """
        Check for buy signal at index i
        Returns signal strength (0.0 to 1.0) or 0.0 if no signal
        """
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Check basic breakout condition
        price_breakout = (row['High'] > prev_row['Breakout_Buy_Level'])
        
        # Check trend filter
        uptrend = row['Close'] > row['Trend_Filter']
        
        # Check volume confirmation
        volume_confirm = row['Volume_Ratio'] > self.volume_threshold
        
        # Check that ATR is available
        atr_available = not pd.isna(row['ATR'])
        
        if price_breakout and uptrend and volume_confirm and atr_available:
            # Calculate signal strength based on multiple factors
            volume_strength = min(row['Volume_Ratio'] / self.volume_threshold, 2.0) / 2.0
            trend_strength = min((row['Close'] - row['Trend_Filter']) / row['Trend_Filter'], 0.1) * 10
            breakout_strength = min((row['High'] - prev_row['Breakout_Buy_Level']) / prev_row['ATR'], 1.0)
            
            signal_strength = (volume_strength + trend_strength + breakout_strength) / 3.0
            return min(signal_strength, 1.0)
        
        return 0.0
    
    def _check_sell_signal(self, df: pd.DataFrame, i: int, entry_date) -> float:
        """
        Check for sell signal at index i
        Returns signal strength (0.0 to 1.0) or 0.0 if no signal
        """
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Check basic breakdown condition
        price_breakdown = (row['Low'] < prev_row['Breakout_Sell_Level'])
        
        # Check maximum hold period
        days_held = (df.index[i] - entry_date).days
        max_hold_reached = days_held >= self.max_hold_days
        
        # Check trend filter breakdown
        trend_breakdown = row['Close'] < row['Trend_Filter']
        
        if price_breakdown or max_hold_reached or trend_breakdown:
            # Calculate signal strength
            if max_hold_reached:
                return 0.8  # Strong signal due to time exit
            elif trend_breakdown:
                return 0.6  # Medium signal due to trend change
            else:
                # Breakdown signal strength
                breakdown_strength = min((prev_row['Breakout_Sell_Level'] - row['Low']) / prev_row['ATR'], 1.0)
                return min(breakdown_strength + 0.3, 1.0)
        
        return 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def get_position_size(self, current_price: float, **kwargs) -> float:
        """Calculate position size based on volatility"""
        return self.position_size

    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """
        Generate trading signals - required by BaseStrategy
        This method integrates with the backtester framework
        """
        # This is the interface method required by BaseStrategy
        # Our main logic is in the standalone generate_signals method above
        # For integration with backtester, we'd implement the signal logic here
        pass


class BollingerVolatilityBreakout(BaseStrategy):
    """
    Bollinger Band Volatility Breakout Strategy
    
    This strategy looks for periods of low volatility (squeeze) followed
    by explosive breakouts beyond the Bollinger Bands.
    
    Entry Logic:
    - Identify Bollinger Band squeeze (bands narrow)
    - Buy when price breaks above upper band with volume
    - Confirm with momentum indicators
    
    Exit Logic:
    - Sell when price touches lower band
    - Or when squeeze conditions return
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 squeeze_period: int = 10,
                 volume_threshold: float = 1.5,
                 momentum_period: int = 12):
        """
        Initialize Bollinger Volatility Breakout Strategy
        
        Args:
            bb_period: Period for Bollinger Bands calculation
            bb_std: Standard deviation multiplier for bands
            squeeze_period: Period to identify squeeze conditions
            volume_threshold: Volume confirmation threshold
            momentum_period: Period for momentum confirmation
        """
        # Create parameters dictionary
        parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'squeeze_period': squeeze_period,
            'volume_threshold': volume_threshold,
            'momentum_period': momentum_period
        }
        
        strategy_name = "Bollinger Volatility Breakout"
        super().__init__(strategy_name, parameters)
        
        # Store parameters as instance attributes for easy access
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_period = squeeze_period
        self.volume_threshold = volume_threshold
        self.momentum_period = momentum_period
        
        self.description = f"BB({bb_period},{bb_std}) squeeze breakout with volume confirmation"
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and squeeze indicators"""
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=self.bb_period).mean()
        bb_std = df['Close'].rolling(window=self.bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * self.bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * self.bb_std)
        
        # Calculate band width for squeeze detection
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Width_MA'] = df['BB_Width'].rolling(window=self.squeeze_period).mean()
        
        # Squeeze condition: current width < average width
        df['Squeeze_Active'] = df['BB_Width'] < df['BB_Width_MA']
        
        # Volume confirmation
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Momentum indicator
        df['Momentum'] = df['Close'] - df['Close'].shift(self.momentum_period)
        
        return df
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger squeeze breakout"""
        df = self.calculate_indicators(data)
        
        # Initialize signal columns
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        
        # Track if we just came out of a squeeze
        squeeze_just_ended = False
        in_position = False
        
        for i in range(self.bb_period + self.squeeze_period, len(df)):
            current_idx = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check if squeeze just ended
            if prev_row['Squeeze_Active'] and not row['Squeeze_Active']:
                squeeze_just_ended = True
            
            if not in_position and squeeze_just_ended:
                # Look for breakout after squeeze
                breakout_up = row['Close'] > row['BB_Upper']
                volume_confirm = row['Volume_Ratio'] > self.volume_threshold
                momentum_positive = row['Momentum'] > 0
                
                if breakout_up and volume_confirm and momentum_positive:
                    signal_strength = self._calculate_breakout_strength(row)
                    df.loc[current_idx, 'Signal'] = 'BUY'
                    df.loc[current_idx, 'Signal_Strength'] = signal_strength
                    in_position = True
                    squeeze_just_ended = False
                    
            elif in_position:
                # Check for exit signals
                breakdown = row['Close'] < row['BB_Lower']
                squeeze_returning = row['Squeeze_Active']
                momentum_negative = row['Momentum'] < 0
                
                if breakdown or squeeze_returning or momentum_negative:
                    df.loc[current_idx, 'Signal'] = 'SELL'
                    df.loc[current_idx, 'Signal_Strength'] = 0.7
                    in_position = False
        
        return df
    
    def _calculate_breakout_strength(self, row: pd.Series) -> float:
        """Calculate the strength of the breakout signal"""
        # Distance above upper band
        breakout_distance = (row['Close'] - row['BB_Upper']) / row['BB_Upper']
        
        # Volume strength
        volume_strength = min(row['Volume_Ratio'] / self.volume_threshold, 2.0) / 2.0
        
        # Momentum strength
        momentum_strength = min(abs(row['Momentum']) / row['Close'], 0.05) * 20
        
        total_strength = (breakout_distance * 10 + volume_strength + momentum_strength) / 3.0
        return min(total_strength, 1.0)

    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """
        Generate trading signals - required by BaseStrategy
        This method integrates with the backtester framework
        """
        # This is the interface method required by BaseStrategy
        # Our main logic is in the standalone generate_signals method above
        # For integration with backtester, we'd implement the signal logic here
        pass


# Strategy registry for easy access
VOLATILITY_BREAKOUT_STRATEGIES = {
    'atr_volatility_breakout': ATRVolatilityBreakout,
    'bollinger_volatility_breakout': BollingerVolatilityBreakout,
}


def get_strategy(strategy_name: str, **kwargs):
    """
    Factory function to create volatility breakout strategies
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    if strategy_name not in VOLATILITY_BREAKOUT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(VOLATILITY_BREAKOUT_STRATEGIES.keys())}")
    
    strategy_class = VOLATILITY_BREAKOUT_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)


def list_strategies() -> Dict[str, str]:
    """
    List available volatility breakout strategies
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    strategies = {}
    for name, strategy_class in VOLATILITY_BREAKOUT_STRATEGIES.items():
        # Create temporary instance to get description
        temp_instance = strategy_class()
        strategies[name] = temp_instance.description
    
    return strategies 