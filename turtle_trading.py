"""
Turtle Trading System Implementation
Classic breakout strategy based on the famous Turtle Trading rules
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from ..indicators.breakout import rolling_high, rolling_low
from ..indicators.volatility import atr


class TurtleTradingStrategy(BaseStrategy):
    """
    Turtle Trading System
    
    Classic trend-following breakout strategy based on the original Turtle Trading rules:
    - Entry: 20-day breakout (System 1) or 55-day breakout (System 2)
    - Exit: 10-day breakout (System 1) or 20-day breakout (System 2) in opposite direction
    - Position sizing: Based on N (20-day ATR)
    - Risk management: 2% risk per trade, maximum 4 units per market
    """
    
    def __init__(self, system: int = 1, variant: str = 'standard'):
        system_name = f"System {system}"
        super().__init__(f"Turtle Trading {system_name} {variant.title()}")
        self.system = system
        self.variant = variant
        self.description = f"Turtle Trading {system_name} - {variant} variant"
        
        # System parameters
        if system == 1:
            # System 1: Shorter-term system
            base_params = {
                'entry_period': 20,
                'exit_period': 10,
                'atr_period': 20,
                'risk_per_trade': 0.02,  # 2% risk per trade
                'max_units': 4,          # Maximum 4 units per market
                'unit_spacing': 0.5,     # Add units every 0.5N
                'stop_loss_n': 2.0       # Stop loss at 2N
            }
        else:
            # System 2: Longer-term system
            base_params = {
                'entry_period': 55,
                'exit_period': 20,
                'atr_period': 20,
                'risk_per_trade': 0.02,
                'max_units': 4,
                'unit_spacing': 0.5,
                'stop_loss_n': 2.0
            }
        
        # Variant modifications
        if variant == 'conservative':
            base_params.update({
                'risk_per_trade': 0.01,  # 1% risk
                'max_units': 2,          # Maximum 2 units
                'stop_loss_n': 1.5       # Tighter stop
            })
        elif variant == 'aggressive':
            base_params.update({
                'risk_per_trade': 0.03,  # 3% risk
                'max_units': 6,          # Maximum 6 units
                'stop_loss_n': 2.5       # Wider stop
            })
        elif variant == 'simple':
            base_params.update({
                'max_units': 1,          # Single unit only
                'unit_spacing': 999,     # No additional units
            })
        
        self.parameters = base_params
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Turtle Trading signals"""
        # Calculate indicators
        entry_high = rolling_high(data['high'], self.parameters['entry_period'])
        entry_low = rolling_low(data['low'], self.parameters['entry_period'])
        exit_high = rolling_high(data['high'], self.parameters['exit_period'])
        exit_low = rolling_low(data['low'], self.parameters['exit_period'])
        
        # Calculate N (True Range)
        n_value = atr(data['high'], data['low'], data['close'], self.parameters['atr_period'])
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Entry conditions
        long_entry = data['high'] > entry_high.shift(1)
        short_entry = data['low'] < entry_low.shift(1)
        
        # Exit conditions
        long_exit = data['low'] < exit_low.shift(1)
        short_exit = data['high'] > exit_high.shift(1)
        
        # Generate signals
        signals.loc[long_entry] = 'BUY'
        signals.loc[short_entry] = 'SELL'
        signals.loc[long_exit] = 'CLOSE_LONG'
        signals.loc[short_exit] = 'CLOSE_SHORT'
        
        return signals
    
    def calculate_position_size(self, data: pd.DataFrame, current_price: float, 
                              portfolio_value: float, current_index: int) -> int:
        """
        Calculate Turtle position size based on N and risk management
        
        Args:
            data: Price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            current_index: Current data index
            
        Returns:
            Number of shares/contracts to trade
        """
        if current_index < self.parameters['atr_period']:
            return 0
        
        # Calculate N (ATR)
        n_value = atr(
            data['high'].iloc[:current_index+1], 
            data['low'].iloc[:current_index+1], 
            data['close'].iloc[:current_index+1], 
            self.parameters['atr_period']
        ).iloc[-1]
        
        if n_value == 0 or np.isnan(n_value):
            return 0
        
        # Calculate dollar volatility (N * price per point)
        dollar_volatility = n_value * 1  # Assuming 1 point = $1
        
        # Calculate unit size
        risk_amount = portfolio_value * self.parameters['risk_per_trade']
        unit_size = int(risk_amount / dollar_volatility)
        
        return max(unit_size, 0)
    
    def get_stop_loss_price(self, entry_price: float, n_value: float, 
                           position_type: str) -> float:
        """
        Calculate stop loss price based on N
        
        Args:
            entry_price: Entry price
            n_value: Current N (ATR) value
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Stop loss price
        """
        stop_distance = n_value * self.parameters['stop_loss_n']
        
        if position_type == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance


class TurtleTradingSystem1(TurtleTradingStrategy):
    """Turtle Trading System 1 (20-day breakout)"""
    
    def __init__(self, variant: str = 'standard'):
        super().__init__(system=1, variant=variant)


class TurtleTradingSystem2(TurtleTradingStrategy):
    """Turtle Trading System 2 (55-day breakout)"""
    
    def __init__(self, variant: str = 'standard'):
        super().__init__(system=2, variant=variant)


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy
    
    Simplified version of Turtle Trading focusing on pure breakout signals
    """
    
    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        super().__init__(f"Donchian Breakout {entry_period}/{exit_period}")
        self.description = f"Donchian Channel breakout strategy ({entry_period}/{exit_period} periods)"
        
        self.parameters = {
            'entry_period': entry_period,
            'exit_period': exit_period,
            'atr_period': 14,
            'min_atr_filter': True,  # Only trade when ATR > recent average
            'trend_filter': False    # Optional trend filter
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Donchian breakout signals"""
        # Calculate Donchian channels
        entry_high = rolling_high(data['high'], self.parameters['entry_period'])
        entry_low = rolling_low(data['low'], self.parameters['entry_period'])
        exit_high = rolling_high(data['high'], self.parameters['exit_period'])
        exit_low = rolling_low(data['low'], self.parameters['exit_period'])
        
        # Optional ATR filter
        if self.parameters['min_atr_filter']:
            current_atr = atr(data['high'], data['low'], data['close'], self.parameters['atr_period'])
            avg_atr = current_atr.rolling(50).mean()
            atr_filter = current_atr > avg_atr
        else:
            atr_filter = pd.Series(True, index=data.index)
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Entry conditions
        long_entry = (data['high'] > entry_high.shift(1)) & atr_filter
        short_entry = (data['low'] < entry_low.shift(1)) & atr_filter
        
        # Exit conditions
        long_exit = data['low'] < exit_low.shift(1)
        short_exit = data['high'] > exit_high.shift(1)
        
        # Generate signals
        signals.loc[long_entry] = 'BUY'
        signals.loc[short_entry] = 'SELL'
        signals.loc[long_exit] = 'CLOSE_LONG'
        signals.loc[short_exit] = 'CLOSE_SHORT'
        
        return signals


# Strategy Registry for Turtle Trading
TURTLE_STRATEGIES = {
    # Turtle Trading System 1 (20-day)
    'turtle_system1_standard': lambda: TurtleTradingSystem1('standard'),
    'turtle_system1_conservative': lambda: TurtleTradingSystem1('conservative'),
    'turtle_system1_aggressive': lambda: TurtleTradingSystem1('aggressive'),
    'turtle_system1_simple': lambda: TurtleTradingSystem1('simple'),
    
    # Turtle Trading System 2 (55-day)
    'turtle_system2_standard': lambda: TurtleTradingSystem2('standard'),
    'turtle_system2_conservative': lambda: TurtleTradingSystem2('conservative'),
    'turtle_system2_aggressive': lambda: TurtleTradingSystem2('aggressive'),
    'turtle_system2_simple': lambda: TurtleTradingSystem2('simple'),
    
    # Donchian Breakout variants
    'donchian_20_10': lambda: DonchianBreakoutStrategy(20, 10),
    'donchian_55_20': lambda: DonchianBreakoutStrategy(55, 20),
    'donchian_10_5': lambda: DonchianBreakoutStrategy(10, 5),
    'donchian_30_15': lambda: DonchianBreakoutStrategy(30, 15),
}


def get_turtle_strategy(strategy_name: str) -> BaseStrategy:
    """Get a turtle trading strategy by name"""
    if strategy_name not in TURTLE_STRATEGIES:
        available = list(TURTLE_STRATEGIES.keys())
        raise ValueError(f"Turtle strategy '{strategy_name}' not found. Available: {available}")
    
    return TURTLE_STRATEGIES[strategy_name]()


def list_turtle_strategies() -> list:
    """List all available turtle trading strategies"""
    return list(TURTLE_STRATEGIES.keys()) 