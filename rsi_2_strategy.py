"""
2-Period RSI Strategy - Larry Connor's Famous Mean Reversion Strategy

This is one of the most well-known quantitative trading strategies.
It buys when RSI(2) is extremely oversold and the stock is in a long-term uptrend.

Original Rules:
- Entry: RSI(2) < 2 AND Price > SMA(200)
- Exit: Close > SMA(5)

Enhanced Version includes:
- Volume confirmation
- Multiple exit conditions
- Risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, StrategyParameters

# Import indicators
from ..indicators.momentum import rsi
from ..indicators.basic_trend import sma
from ..indicators.volume import volume_ratio


class RSI2Strategy(BaseStrategy):
    """
    2-Period RSI Strategy Class
    
    Larry Connor's famous mean reversion strategy that buys extremely
    oversold stocks that are in long-term uptrends.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize 2-Period RSI Strategy
        
        Parameters:
            rsi_period: RSI calculation period (default: 2)
            rsi_entry_threshold: RSI level for entry (default: 2)
            rsi_exit_threshold: RSI level for exit (default: 95)
            trend_sma_period: Long-term trend SMA (default: 200)
            exit_sma_period: Exit SMA period (default: 5)
            volume_confirmation: Require volume confirmation (default: True)
            volume_threshold: Volume ratio threshold (default: 1.2)
        """
        default_params = {
            'rsi_period': 2,
            'rsi_entry_threshold': 2,
            'rsi_exit_threshold': 95,
            'trend_sma_period': 200,
            'exit_sma_period': 5,
            'volume_confirmation': True,
            'volume_threshold': 1.2,
            'max_holding_days': 10  # Maximum days to hold position
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("2-Period RSI Strategy", default_params)
        self.params = StrategyParameters(**self.parameters)
    
    def get_required_lookback(self) -> int:
        """Return required lookback period"""
        return max(self.params.trend_sma_period + 50, 250)
    
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """
        Generate RSI-2 trading signals
        
        Args:
            backtester: Backtester instance
            daily_data: Current day's market data
        """
        for _, row in daily_data.iterrows():
            symbol = row['Symbol']
            
            # Get historical data
            hist_data = backtester.get_current_data(symbol, lookback=self.get_required_lookback())
            
            if hist_data.empty or len(hist_data) < self.params.trend_sma_period:
                continue
            
            # Calculate indicators
            rsi_values = rsi(hist_data['Close'], self.params.rsi_period)
            trend_sma = sma(hist_data['Close'], self.params.trend_sma_period)
            exit_sma = sma(hist_data['Close'], self.params.exit_sma_period)
            
            if self.params.volume_confirmation:
                vol_ratio = volume_ratio(hist_data['Volume'], 20)
            
            # Current values
            current_price = hist_data['Close'].iloc[-1]
            current_rsi = rsi_values.iloc[-1]
            current_trend_sma = trend_sma.iloc[-1]
            current_exit_sma = exit_sma.iloc[-1]
            
            if self.params.volume_confirmation:
                current_vol_ratio = vol_ratio.iloc[-1]
            
            # Check current position
            position = backtester.portfolio.get_position(symbol)
            
            # Exit conditions (check first)
            if position and position.quantity > 0:
                should_exit, exit_reason = self.should_exit_trade(
                    backtester, symbol, hist_data, current_price, 
                    current_rsi, current_exit_sma, position
                )
                
                if should_exit:
                    self.execute_signal(backtester, symbol, 'CLOSE', exit_reason)
                    continue
            
            # Entry conditions
            if not position or position.quantity == 0:
                should_enter, entry_reason = self.should_enter_trade(
                    backtester, symbol, hist_data, current_price, current_rsi, 
                    current_trend_sma, current_vol_ratio if self.params.volume_confirmation else None
                )
                
                if should_enter:
                    self.execute_signal(backtester, symbol, 'BUY', entry_reason)
    
    def should_enter_trade(self, backtester, symbol: str, hist_data: pd.DataFrame, 
                          current_price: float, current_rsi: float, current_trend_sma: float,
                          current_vol_ratio: Optional[float] = None) -> tuple[bool, str]:
        """
        Check RSI-2 entry conditions
        
        Returns:
            (should_enter, reason)
        """
        # Core RSI-2 conditions
        rsi_oversold = current_rsi < self.params.rsi_entry_threshold
        in_uptrend = current_price > current_trend_sma
        
        # Volume confirmation (if enabled)
        volume_ok = True
        if self.params.volume_confirmation and current_vol_ratio is not None:
            volume_ok = current_vol_ratio > self.params.volume_threshold
        
        # Additional filters
        price_valid = current_price > 5.0  # Avoid penny stocks
        
        if rsi_oversold and in_uptrend and volume_ok and price_valid:
            reason_parts = [f"RSI({self.params.rsi_period})={current_rsi:.1f}<{self.params.rsi_entry_threshold}"]
            reason_parts.append(f"Price>${current_price:.2f}>SMA({self.params.trend_sma_period})${current_trend_sma:.2f}")
            
            if self.params.volume_confirmation:
                reason_parts.append(f"Volume={current_vol_ratio:.1f}x>{self.params.volume_threshold}")
            
            return True, " & ".join(reason_parts)
        
        return False, "Entry conditions not met"
    
    def should_exit_trade(self, backtester, symbol: str, hist_data: pd.DataFrame,
                         current_price: float, current_rsi: float, current_exit_sma: float,
                         position) -> tuple[bool, str]:
        """
        Check RSI-2 exit conditions
        
        Returns:
            (should_exit, reason)
        """
        # Primary exit: Price above exit SMA
        if current_price > current_exit_sma:
            return True, f"Price${current_price:.2f}>SMA({self.params.exit_sma_period})${current_exit_sma:.2f}"
        
        # Secondary exit: RSI extremely overbought
        if current_rsi > self.params.rsi_exit_threshold:
            return True, f"RSI({self.params.rsi_period})={current_rsi:.1f}>{self.params.rsi_exit_threshold}"
        
        # Time-based exit: Maximum holding period
        if hasattr(position, 'entry_date') and backtester.current_date:
            days_held = (backtester.current_date - position.entry_date).days
            if days_held >= self.params.max_holding_days:
                return True, f"Max holding period {self.params.max_holding_days} days reached"
        
        return False, "No exit conditions met"


# Functional interface for backward compatibility
def rsi_2_strategy(backtester, daily_data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None):
    """
    Functional interface for 2-Period RSI Strategy
    
    Args:
        backtester: Backtester instance
        daily_data: Current day's market data
        parameters: Strategy parameters (optional)
    """
    strategy = RSI2Strategy(parameters)
    strategy.generate_signals(backtester, daily_data)


# Strategy variants
def rsi_2_conservative(backtester, daily_data: pd.DataFrame):
    """Conservative version with stricter entry conditions"""
    params = {
        'rsi_entry_threshold': 1,  # Even more oversold
        'volume_threshold': 1.5,   # Higher volume requirement
        'trend_sma_period': 200,   # Keep long-term trend
    }
    rsi_2_strategy(backtester, daily_data, params)


def rsi_2_aggressive(backtester, daily_data: pd.DataFrame):
    """Aggressive version with looser entry conditions"""
    params = {
        'rsi_entry_threshold': 5,  # Less oversold
        'volume_threshold': 1.0,   # Lower volume requirement
        'trend_sma_period': 100,   # Shorter trend period
        'max_holding_days': 15,    # Longer holding period
    }
    rsi_2_strategy(backtester, daily_data, params)


def rsi_2_no_volume(backtester, daily_data: pd.DataFrame):
    """Version without volume confirmation"""
    params = {
        'volume_confirmation': False,
    }
    rsi_2_strategy(backtester, daily_data, params)


# Export strategy information
STRATEGY_INFO = {
    'name': '2-Period RSI Strategy',
    'author': 'Larry Connor',
    'type': 'Mean Reversion',
    'timeframe': 'Daily',
    'description': 'Buys extremely oversold stocks in uptrends, exits on recovery',
    'typical_holding_period': '1-5 days',
    'win_rate': '60-70%',
    'best_markets': 'Trending bull markets',
    'parameters': {
        'rsi_period': 'RSI calculation period (default: 2)',
        'rsi_entry_threshold': 'RSI level for entry (default: 2)',
        'rsi_exit_threshold': 'RSI level for exit (default: 95)',
        'trend_sma_period': 'Long-term trend SMA (default: 200)',
        'exit_sma_period': 'Exit SMA period (default: 5)',
        'volume_confirmation': 'Require volume confirmation (default: True)',
        'volume_threshold': 'Volume ratio threshold (default: 1.2)',
        'max_holding_days': 'Maximum days to hold position (default: 10)'
    }
} 