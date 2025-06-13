"""
Strategies Module - Complete Trading Strategy Collection
Comprehensive collection of proven quantitative trading strategies
"""

from .base_strategy import BaseStrategy
from .rsi_2_strategy import RSI2Strategy, rsi_2_strategy
from .double_7s_strategy import Double7sStrategy, double_7s_strategy
from .strategy_comparison import StrategyComparison, compare_strategies

# Import Larry Connor strategies
from .larry_connor_strategies import (
    LARRY_CONNOR_STRATEGIES, get_strategy as get_larry_connor_strategy,
    list_strategies as list_larry_connor_strategies
)

# Import Turtle Trading strategies
from .turtle_trading import (
    TURTLE_STRATEGIES, get_turtle_strategy, list_turtle_strategies
)

# Import Trend Following strategies
from .trend_following import (
    TREND_FOLLOWING_STRATEGIES, get_strategy as get_trend_strategy
)

__all__ = [
    'BaseStrategy',
    'RSI2Strategy', 
    'rsi_2_strategy',
    'Double7sStrategy',
    'double_7s_strategy', 
    'StrategyComparison',
    'compare_strategies',
    'LARRY_CONNOR_STRATEGIES',
    'TURTLE_STRATEGIES', 
    'TREND_FOLLOWING_STRATEGIES',
    'get_larry_connor_strategy',
    'get_turtle_strategy',
    'get_trend_strategy',
    'list_larry_connor_strategies',
    'list_turtle_strategies',
    'get_strategy',
    'list_strategies'
]

# Combined strategy registry for easy access
ALL_STRATEGIES = {
    **LARRY_CONNOR_STRATEGIES,
    **TURTLE_STRATEGIES,
    **TREND_FOLLOWING_STRATEGIES,
    # Legacy strategies
    'rsi_2': rsi_2_strategy,
    'double_7s': double_7s_strategy,
}

def get_strategy(name: str):
    """Get strategy by name from all available strategies"""
    if name in ALL_STRATEGIES:
        return ALL_STRATEGIES[name]()
    return None

def list_strategies():
    """List all available strategies"""
    return list(ALL_STRATEGIES.keys())

def list_strategies_by_category():
    """List strategies organized by category"""
    return {
        'Larry Connor': list(LARRY_CONNOR_STRATEGIES.keys()),
        'Turtle Trading': list(TURTLE_STRATEGIES.keys()),
        'Trend Following': list(TREND_FOLLOWING_STRATEGIES.keys()),
        'Legacy': ['rsi_2', 'double_7s']
    } 