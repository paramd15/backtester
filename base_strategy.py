"""
Base Strategy Class
Provides common functionality for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    Provides common functionality:
    - Parameter management
    - Performance tracking
    - Signal generation framework
    - Risk management hooks
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            parameters: Strategy parameters dictionary
        """
        self.name = name
        self.parameters = parameters or {}
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_signal_date = None
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """
        Generate trading signals - must be implemented by each strategy
        
        Args:
            backtester: Backtester instance
            daily_data: Current day's market data
        """
        pass
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        return True
    
    def get_required_lookback(self) -> int:
        """Get minimum lookback period required for indicators"""
        return 250  # Default to 1 year
    
    def calculate_position_size(self, backtester, symbol: str, signal_strength: float = 1.0) -> int:
        """
        Calculate position size with strategy-specific logic
        
        Args:
            backtester: Backtester instance
            symbol: Trading symbol
            signal_strength: Signal strength (0.0 to 1.0)
            
        Returns:
            Position size in shares
        """
        # Use backtester's default position sizing
        return backtester.calculate_position_size(symbol, signal_strength)
    
    def should_enter_trade(self, backtester, symbol: str, hist_data: pd.DataFrame) -> tuple[bool, str]:
        """
        Check if we should enter a trade
        
        Returns:
            (should_enter, reason)
        """
        return False, "No entry conditions met"
    
    def should_exit_trade(self, backtester, symbol: str, hist_data: pd.DataFrame) -> tuple[bool, str]:
        """
        Check if we should exit a trade
        
        Returns:
            (should_exit, reason)
        """
        return False, "No exit conditions met"
    
    def execute_signal(self, backtester, symbol: str, signal: str, reason: str = "") -> bool:
        """
        Execute a trading signal with logging
        
        Args:
            backtester: Backtester instance
            symbol: Trading symbol
            signal: Signal type (BUY, SELL, CLOSE)
            reason: Reason for the signal
            
        Returns:
            True if signal was executed successfully
        """
        success = backtester.execute_signal(symbol, signal)
        
        if success:
            self.trades_executed += 1
            self.last_signal_date = backtester.current_date
        
        self.signals_generated += 1
        return success
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'last_signal_date': self.last_signal_date,
            'execution_rate': self.trades_executed / max(1, self.signals_generated),
            'required_lookback': self.get_required_lookback()
        }
    
    def reset_stats(self):
        """Reset strategy statistics"""
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_signal_date = None
        self.performance_metrics = {}
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class StrategyParameters:
    """Helper class for managing strategy parameters"""
    
    def __init__(self, **kwargs):
        """Initialize with parameter values"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def update(self, **kwargs):
        """Update parameter values"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __str__(self) -> str:
        params = ', '.join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"Parameters({params})" 