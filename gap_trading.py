"""
Gap Trading Strategies
Implementation of Phase 1.2 from roadmap - Classic Breakout Strategies

This module implements gap-based trading strategies that capitalize on
opening price gaps and their subsequent momentum or mean reversion patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_strategy import BaseStrategy


class GapUpMomentumStrategy(BaseStrategy):
    """
    Gap Up Momentum Strategy
    
    Entry Logic:
    - Identify significant gap ups (open > previous close by X%)
    - Must have volume confirmation (volume > threshold * average)
    - Gap size must be within reasonable range (not too large)
    - Enter at market open or on first pullback
    
    Exit Logic:
    - Hold for predetermined period or until stop loss hit
    - Exit on gap fill (price returns to previous close)
    - Exit on volume exhaustion
    
    This strategy captures momentum continuation after positive gaps.
    """
    
    def __init__(self, 
                 min_gap_percent: float = 2.0,
                 max_gap_percent: float = 8.0,
                 volume_threshold: float = 2.0,
                 hold_days: int = 3,
                 stop_loss_percent: float = 5.0,
                 position_size: float = 0.1):
        """
        Initialize Gap Up Momentum Strategy
        
        Args:
            min_gap_percent: Minimum gap size to trigger signal
            max_gap_percent: Maximum gap size (avoid runaway gaps)
            volume_threshold: Volume must be X times average
            hold_days: Maximum days to hold position
            stop_loss_percent: Stop loss percentage
            position_size: Fraction of capital to use per trade
        """
        parameters = {
            'min_gap_percent': min_gap_percent,
            'max_gap_percent': max_gap_percent,
            'volume_threshold': volume_threshold,
            'hold_days': hold_days,
            'stop_loss_percent': stop_loss_percent,
            'position_size': position_size
        }
        
        super().__init__("Gap Up Momentum", parameters)
        
        self.min_gap_percent = min_gap_percent
        self.max_gap_percent = max_gap_percent
        self.volume_threshold = volume_threshold
        self.hold_days = hold_days
        self.stop_loss_percent = stop_loss_percent
        self.position_size = position_size
        
        self.description = f"Gap up {min_gap_percent}%-{max_gap_percent}% momentum with {hold_days}d hold"
        
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """Generate trading signals - required by BaseStrategy"""
        pass
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate gap momentum signals"""
        df = data.copy()
        
        # Calculate gap metrics
        df['Prev_Close'] = df['Close'].shift(1)
        df['Gap_Percent'] = ((df['Open'] - df['Prev_Close']) / df['Prev_Close']) * 100
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Initialize signals
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        # Generate signals
        in_position = False
        entry_date = None
        entry_price = None
        
        for i in range(1, len(df)):
            if not in_position:
                gap_ok = self.min_gap_percent <= df.iloc[i]['Gap_Percent'] <= self.max_gap_percent
                volume_ok = df.iloc[i]['Volume_Ratio'] > self.volume_threshold
                
                if gap_ok and volume_ok:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.8
                    df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Open']
                    in_position = True
                    entry_date = df.index[i]
                    entry_price = df.iloc[i]['Open']
            else:
                days_held = (df.index[i] - entry_date).days
                loss_pct = ((df.iloc[i]['Close'] - entry_price) / entry_price) * 100
                
                if days_held >= self.hold_days or loss_pct <= -self.stop_loss_percent:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = df.iloc[i]['Close']
                    in_position = False
                    entry_date = None
                    entry_price = None
        
        return df


class GapFadeStrategy(BaseStrategy):
    """
    Gap Fade Strategy (Mean Reversion)
    
    Entry Logic:
    - Identify large gaps (usually gap downs or very large gap ups)
    - Enter position expecting gap to fill (mean reversion)
    - Wait for initial momentum to exhaust
    
    Exit Logic:
    - Exit when gap is partially or fully filled
    - Exit on profit target reached
    - Exit if gap continues (stop loss)
    
    This strategy captures mean reversion after excessive gaps.
    """
    
    def __init__(self,
                 min_gap_percent: float = 3.0,
                 fade_threshold: float = 5.0,
                 hold_days: int = 2,
                 profit_target_percent: float = 3.0,
                 position_size: float = 0.05):
        """
        Initialize Gap Fade Strategy
        
        Args:
            min_gap_percent: Minimum gap size to consider fading
            fade_threshold: Gap size threshold for aggressive fading
            hold_days: Maximum days to hold position
            profit_target_percent: Profit target percentage
            position_size: Fraction of capital (smaller for higher risk)
        """
        parameters = {
            'min_gap_percent': min_gap_percent,
            'fade_threshold': fade_threshold,
            'hold_days': hold_days,
            'profit_target_percent': profit_target_percent,
            'position_size': position_size
        }
        
        super().__init__("Gap Fade", parameters)
        
        self.min_gap_percent = min_gap_percent
        self.fade_threshold = fade_threshold
        self.hold_days = hold_days
        self.profit_target_percent = profit_target_percent
        self.position_size = position_size
        
        self.description = f"Fade gaps >{min_gap_percent}% expecting {profit_target_percent}% reversion"
        
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """Generate trading signals - required by BaseStrategy"""
        pass
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate gap fade signals"""
        df = data.copy()
        
        # Calculate gap metrics
        df['Prev_Close'] = df['Close'].shift(1)
        df['Gap_Percent'] = ((df['Open'] - df['Prev_Close']) / df['Prev_Close']) * 100
        
        # Initialize signals
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        df['Trade_Direction'] = ''
        
        # Generate signals
        in_position = False
        entry_date = None
        entry_price = None
        trade_direction = None
        
        for i in range(1, len(df)):
            if not in_position:
                # Large gap down - fade it (buy)
                if df.iloc[i]['Gap_Percent'] <= -self.min_gap_percent:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.7
                    df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
                    df.iloc[i, df.columns.get_loc('Trade_Direction')] = 'LONG'
                    in_position = True
                    entry_date = df.index[i]
                    entry_price = df.iloc[i]['Close']
                    trade_direction = 'LONG'
                
                # Very large gap up - fade it (sell)
                elif df.iloc[i]['Gap_Percent'] >= self.fade_threshold:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.6
                    df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
                    df.iloc[i, df.columns.get_loc('Trade_Direction')] = 'SHORT'
                    in_position = True
                    entry_date = df.index[i]
                    entry_price = df.iloc[i]['Close']
                    trade_direction = 'SHORT'
            else:
                days_held = (df.index[i] - entry_date).days
                
                if trade_direction == 'LONG':
                    pnl_pct = ((df.iloc[i]['Close'] - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - df.iloc[i]['Close']) / entry_price) * 100
                
                if days_held >= self.hold_days or pnl_pct >= self.profit_target_percent:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'CLOSE'
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = df.iloc[i]['Close']
                    in_position = False
                    entry_date = None
                    entry_price = None
                    trade_direction = None
        
        return df


# Strategy registry
GAP_TRADING_STRATEGIES = {
    'gap_up_momentum': GapUpMomentumStrategy,
    'gap_fade_strategy': GapFadeStrategy,
}


def get_strategy(strategy_name: str, **kwargs):
    """Factory function to create gap trading strategies"""
    if strategy_name not in GAP_TRADING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = GAP_TRADING_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)


def list_strategies() -> Dict[str, str]:
    """
    List available gap trading strategies
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    strategies = {}
    for name, strategy_class in GAP_TRADING_STRATEGIES.items():
        # Create temporary instance to get description
        temp_instance = strategy_class()
        strategies[name] = temp_instance.description
    
    return strategies 