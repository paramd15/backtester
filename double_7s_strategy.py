"""
Double 7's Strategy - Larry Connor's Mean Reversion Strategy

This strategy combines price action (7-day lows) with momentum (RSI) 
to identify oversold conditions in trending stocks.

Original Rules:
- Entry: Price makes 7-day low AND RSI(2) < 5 AND Price > SMA(200)
- Exit: Price makes 7-day high

Enhanced Version includes:
- Volume confirmation
- Multiple timeframes
- Risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, StrategyParameters

# Import indicators
from ..indicators.momentum import rsi
from ..indicators.basic_trend import sma
from ..indicators.breakout import rolling_high, rolling_low
from ..indicators.volume import volume_ratio


class Double7sStrategy(BaseStrategy):
    """
    Double 7's Strategy Class
    
    Combines 7-day price lows with RSI oversold conditions
    to identify mean reversion opportunities.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Double 7's Strategy
        
        Parameters:
            price_lookback: Days for price high/low (default: 7)
            rsi_period: RSI calculation period (default: 2)
            rsi_threshold: RSI threshold for entry (default: 5)
            trend_sma_period: Long-term trend SMA (default: 200)
            volume_confirmation: Require volume confirmation (default: True)
            volume_threshold: Volume ratio threshold (default: 1.1)
            max_holding_days: Maximum days to hold position (default: 15)
        """
        default_params = {
            'price_lookback': 7,
            'rsi_period': 2,
            'rsi_threshold': 5,
            'trend_sma_period': 200,
            'volume_confirmation': True,
            'volume_threshold': 1.1,
            'max_holding_days': 15,
            'min_price': 5.0  # Avoid penny stocks
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Double 7's Strategy", default_params)
        self.params = StrategyParameters(**self.parameters)
    
    def get_required_lookback(self) -> int:
        """Return required lookback period"""
        return max(self.params.trend_sma_period + 50, 250)
    
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """
        Generate Double 7's trading signals
        
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
            high_7 = rolling_high(hist_data['High'], self.params.price_lookback)
            low_7 = rolling_low(hist_data['Low'], self.params.price_lookback)
            
            if self.params.volume_confirmation:
                vol_ratio = volume_ratio(hist_data['Volume'], 20)
            
            # Current values
            current_price = hist_data['Close'].iloc[-1]
            current_low = hist_data['Low'].iloc[-1]
            current_high = hist_data['High'].iloc[-1]
            current_rsi = rsi_values.iloc[-1]
            current_trend_sma = trend_sma.iloc[-1]
            
            # Get 7-day levels (use -2 to avoid look-ahead bias)
            seven_day_low = low_7.iloc[-2] if len(low_7) > 1 else low_7.iloc[-1]
            seven_day_high = high_7.iloc[-2] if len(high_7) > 1 else high_7.iloc[-1]
            
            if self.params.volume_confirmation:
                current_vol_ratio = vol_ratio.iloc[-1]
            
            # Check current position
            position = backtester.portfolio.get_position(symbol)
            
            # Exit conditions (check first)
            if position and position.quantity > 0:
                should_exit, exit_reason = self.should_exit_trade(
                    backtester, symbol, hist_data, current_high, 
                    seven_day_high, position
                )
                
                if should_exit:
                    self.execute_signal(backtester, symbol, 'CLOSE', exit_reason)
                    continue
            
            # Entry conditions
            if not position or position.quantity == 0:
                should_enter, entry_reason = self.should_enter_trade(
                    backtester, symbol, hist_data, current_low, current_price,
                    current_rsi, current_trend_sma, seven_day_low,
                    current_vol_ratio if self.params.volume_confirmation else None
                )
                
                if should_enter:
                    self.execute_signal(backtester, symbol, 'BUY', entry_reason)
    
    def should_enter_trade(self, backtester, symbol: str, hist_data: pd.DataFrame,
                          current_low: float, current_price: float, current_rsi: float,
                          current_trend_sma: float, seven_day_low: float,
                          current_vol_ratio: Optional[float] = None) -> tuple[bool, str]:
        """
        Check Double 7's entry conditions
        
        Returns:
            (should_enter, reason)
        """
        # Core Double 7's conditions
        price_at_7day_low = current_low <= seven_day_low
        rsi_oversold = current_rsi < self.params.rsi_threshold
        in_uptrend = current_price > current_trend_sma
        
        # Volume confirmation (if enabled)
        volume_ok = True
        if self.params.volume_confirmation and current_vol_ratio is not None:
            volume_ok = current_vol_ratio > self.params.volume_threshold
        
        # Additional filters
        price_valid = current_price > self.params.min_price
        
        if price_at_7day_low and rsi_oversold and in_uptrend and volume_ok and price_valid:
            reason_parts = [f"7-day low: ${current_low:.2f}<=${seven_day_low:.2f}"]
            reason_parts.append(f"RSI({self.params.rsi_period})={current_rsi:.1f}<{self.params.rsi_threshold}")
            reason_parts.append(f"Price${current_price:.2f}>SMA({self.params.trend_sma_period})${current_trend_sma:.2f}")
            
            if self.params.volume_confirmation:
                reason_parts.append(f"Volume={current_vol_ratio:.1f}x>{self.params.volume_threshold}")
            
            return True, " & ".join(reason_parts)
        
        return False, "Entry conditions not met"
    
    def should_exit_trade(self, backtester, symbol: str, hist_data: pd.DataFrame,
                         current_high: float, seven_day_high: float, position) -> tuple[bool, str]:
        """
        Check Double 7's exit conditions
        
        Returns:
            (should_exit, reason)
        """
        # Primary exit: Price makes 7-day high
        if current_high >= seven_day_high:
            return True, f"7-day high: ${current_high:.2f}>=${seven_day_high:.2f}"
        
        # Time-based exit: Maximum holding period
        if hasattr(position, 'entry_date') and backtester.current_date:
            days_held = (backtester.current_date - position.entry_date).days
            if days_held >= self.params.max_holding_days:
                return True, f"Max holding period {self.params.max_holding_days} days reached"
        
        return False, "No exit conditions met"


# Functional interface for backward compatibility
def double_7s_strategy(backtester, daily_data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None):
    """
    Functional interface for Double 7's Strategy
    
    Args:
        backtester: Backtester instance
        daily_data: Current day's market data
        parameters: Strategy parameters (optional)
    """
    strategy = Double7sStrategy(parameters)
    strategy.generate_signals(backtester, daily_data)


# Strategy variants
def double_7s_conservative(backtester, daily_data: pd.DataFrame):
    """Conservative version with stricter conditions"""
    params = {
        'rsi_threshold': 3,        # More oversold
        'volume_threshold': 1.3,   # Higher volume requirement
        'price_lookback': 10,      # Longer lookback period
    }
    double_7s_strategy(backtester, daily_data, params)


def double_7s_aggressive(backtester, daily_data: pd.DataFrame):
    """Aggressive version with looser conditions"""
    params = {
        'rsi_threshold': 10,       # Less oversold
        'volume_threshold': 1.0,   # Lower volume requirement
        'price_lookback': 5,       # Shorter lookback period
        'max_holding_days': 20,    # Longer holding period
    }
    double_7s_strategy(backtester, daily_data, params)


def double_10s_strategy(backtester, daily_data: pd.DataFrame):
    """10-day version instead of 7-day"""
    params = {
        'price_lookback': 10,
        'rsi_threshold': 5,
        'max_holding_days': 20,
    }
    double_7s_strategy(backtester, daily_data, params)


# Export strategy information
STRATEGY_INFO = {
    'name': 'Double 7s Strategy',
    'author': 'Larry Connor',
    'type': 'Mean Reversion',
    'timeframe': 'Daily',
    'description': 'Buys 7-day lows with RSI confirmation in uptrends',
    'typical_holding_period': '3-10 days',
    'win_rate': '55-65%',
    'best_markets': 'Trending markets with volatility',
    'parameters': {
        'price_lookback': 'Days for price high/low (default: 7)',
        'rsi_period': 'RSI calculation period (default: 2)',
        'rsi_threshold': 'RSI threshold for entry (default: 5)',
        'trend_sma_period': 'Long-term trend SMA (default: 200)',
        'volume_confirmation': 'Require volume confirmation (default: True)',
        'volume_threshold': 'Volume ratio threshold (default: 1.1)',
        'max_holding_days': 'Maximum days to hold position (default: 15)',
        'min_price': 'Minimum stock price (default: 5.0)'
    }
} 