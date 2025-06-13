"""
Larry Connor Trading Strategies
Comprehensive collection of Larry Connor's proven mean reversion strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from ..indicators.momentum import rsi
from ..indicators.basic_trend import sma, ema
from ..indicators.volume import volume_ratio
from ..indicators.volatility import atr


class BaseStrategy(ABC):
    """Base class for all Larry Connor strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.description = ""
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the strategy"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }


class RSI2Strategy(BaseStrategy):
    """
    Larry Connor's 2-Period RSI Strategy (4 variants)
    
    Classic mean reversion strategy using 2-period RSI
    """
    
    def __init__(self, variant: str = 'standard'):
        super().__init__(f"RSI-2 {variant.title()}")
        self.variant = variant
        self.description = f"2-Period RSI mean reversion strategy - {variant} variant"
        
        # Strategy parameters by variant
        if variant == 'standard':
            self.parameters = {
                'rsi_period': 2,
                'rsi_buy_threshold': 2,
                'rsi_sell_threshold': 98,
                'trend_ma_period': 200,
                'exit_ma_period': 5,
                'volume_threshold': 1.0
            }
        elif variant == 'conservative':
            self.parameters = {
                'rsi_period': 2,
                'rsi_buy_threshold': 1,
                'rsi_sell_threshold': 99,
                'trend_ma_period': 200,
                'exit_ma_period': 5,
                'volume_threshold': 1.5
            }
        elif variant == 'aggressive':
            self.parameters = {
                'rsi_period': 2,
                'rsi_buy_threshold': 5,
                'rsi_sell_threshold': 95,
                'trend_ma_period': 100,
                'exit_ma_period': 3,
                'volume_threshold': 0.8
            }
        elif variant == 'no_volume':
            self.parameters = {
                'rsi_period': 2,
                'rsi_buy_threshold': 2,
                'rsi_sell_threshold': 98,
                'trend_ma_period': 200,
                'exit_ma_period': 5,
                'volume_threshold': 0.0  # No volume filter
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI-2 signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        exit_ma = sma(data['close'], self.parameters['exit_ma_period'])
        
        # Volume filter (if applicable)
        if self.parameters['volume_threshold'] > 0:
            vol_ratio = volume_ratio(data['volume'], 20)
            volume_ok = vol_ratio >= self.parameters['volume_threshold']
        else:
            volume_ok = pd.Series(True, index=data.index)
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Entry conditions
        uptrend = data['close'] > trend_ma
        downtrend = data['close'] < trend_ma
        
        # RSI conditions
        rsi_oversold = rsi_2 <= self.parameters['rsi_buy_threshold']
        rsi_overbought = rsi_2 >= self.parameters['rsi_sell_threshold']
        
        # Exit conditions
        above_exit_ma = data['close'] > exit_ma
        below_exit_ma = data['close'] < exit_ma
        
        # Generate signals
        buy_condition = uptrend & rsi_oversold & volume_ok
        sell_condition = downtrend & rsi_overbought & volume_ok
        
        # Exit conditions - restore original logic
        exit_long = above_exit_ma | downtrend
        exit_short = below_exit_ma | uptrend
        
        # Apply signals with proper priority
        signals.loc[buy_condition] = 'BUY'
        signals.loc[sell_condition] = 'SELL'
        
        # Apply exit signals only where we don't have entry signals
        signals.loc[exit_long & ~buy_condition & ~sell_condition] = 'CLOSE_LONG'
        signals.loc[exit_short & ~buy_condition & ~sell_condition] = 'CLOSE_SHORT'
        
        return signals


class Double7Strategy(BaseStrategy):
    """
    Larry Connor's Double 7's Strategy (4 variants)
    
    Mean reversion strategy using 7-day highs/lows with RSI confirmation
    """
    
    def __init__(self, variant: str = 'standard'):
        super().__init__(f"Double 7's {variant.title()}")
        self.variant = variant
        self.description = f"Double 7's mean reversion strategy - {variant} variant"
        
        if variant == 'standard':
            self.parameters = {
                'lookback_period': 7,
                'rsi_period': 2,
                'rsi_threshold': 5,
                'trend_ma_period': 200,
                'volume_threshold': 1.0
            }
        elif variant == 'conservative':
            self.parameters = {
                'lookback_period': 10,
                'rsi_period': 2,
                'rsi_threshold': 3,
                'trend_ma_period': 200,
                'volume_threshold': 1.5
            }
        elif variant == 'aggressive':
            self.parameters = {
                'lookback_period': 5,
                'rsi_period': 2,
                'rsi_threshold': 10,
                'trend_ma_period': 100,
                'volume_threshold': 0.8
            }
        elif variant == 'double_10':
            self.parameters = {
                'lookback_period': 10,
                'rsi_period': 2,
                'rsi_threshold': 5,
                'trend_ma_period': 200,
                'volume_threshold': 1.0
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Double 7's signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        
        # Rolling highs and lows
        rolling_high = data['high'].rolling(self.parameters['lookback_period']).max()
        rolling_low = data['low'].rolling(self.parameters['lookback_period']).min()
        
        # Volume filter
        if self.parameters['volume_threshold'] > 0:
            vol_ratio = volume_ratio(data['volume'], 20)
            volume_ok = vol_ratio >= self.parameters['volume_threshold']
        else:
            volume_ok = pd.Series(True, index=data.index)
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Trend conditions
        uptrend = data['close'] > trend_ma
        downtrend = data['close'] < trend_ma
        
        # Entry conditions
        at_low = data['low'] <= rolling_low.shift(1)  # At or below N-day low
        at_high = data['high'] >= rolling_high.shift(1)  # At or above N-day high
        rsi_oversold = rsi_2 <= self.parameters['rsi_threshold']
        rsi_overbought = rsi_2 >= (100 - self.parameters['rsi_threshold'])
        
        # Exit conditions (opposite extreme)
        exit_long = data['high'] >= rolling_high.shift(1)
        exit_short = data['low'] <= rolling_low.shift(1)
        
        # Generate signals
        buy_condition = uptrend & at_low & rsi_oversold & volume_ok
        sell_condition = downtrend & at_high & rsi_overbought & volume_ok
        
        signals.loc[buy_condition] = 'BUY'
        signals.loc[sell_condition] = 'SELL'
        signals.loc[exit_long & uptrend] = 'CLOSE_LONG'
        signals.loc[exit_short & downtrend] = 'CLOSE_SHORT'
        
        return signals


class RSIPullbackStrategy(BaseStrategy):
    """
    RSI Pullback Strategy
    
    Requires multiple consecutive extreme RSI readings for stronger signals
    """
    
    def __init__(self, consecutive_periods: int = 3):
        super().__init__(f"RSI Pullback ({consecutive_periods} consecutive)")
        self.consecutive_periods = consecutive_periods
        self.description = f"RSI pullback strategy requiring {consecutive_periods} consecutive extreme readings"
        
        self.parameters = {
            'rsi_period': 2,
            'rsi_buy_threshold': 10,
            'rsi_sell_threshold': 90,
            'consecutive_periods': consecutive_periods,
            'trend_ma_period': 200,
            'exit_method': 'previous_high_low'  # Exit when price exceeds previous candle's high/low
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI Pullback signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Trend conditions
        uptrend = data['close'] > trend_ma
        downtrend = data['close'] < trend_ma
        
        # Consecutive RSI conditions
        rsi_oversold = rsi_2 <= self.parameters['rsi_buy_threshold']
        rsi_overbought = rsi_2 >= self.parameters['rsi_sell_threshold']
        
        # Check for consecutive periods
        consecutive_oversold = pd.Series(False, index=data.index)
        consecutive_overbought = pd.Series(False, index=data.index)
        
        for i in range(self.consecutive_periods, len(data)):
            # Check if last N periods were all oversold/overbought
            oversold_count = rsi_oversold.iloc[i-self.consecutive_periods+1:i+1].sum()
            overbought_count = rsi_overbought.iloc[i-self.consecutive_periods+1:i+1].sum()
            
            if oversold_count == self.consecutive_periods:
                consecutive_oversold.iloc[i] = True
            if overbought_count == self.consecutive_periods:
                consecutive_overbought.iloc[i] = True
        
        # Exit conditions (previous candle's high/low)
        prev_high = data['high'].shift(1)
        prev_low = data['low'].shift(1)
        
        # Generate signals
        buy_condition = uptrend & consecutive_oversold
        sell_condition = downtrend & consecutive_overbought
        
        exit_long = (data['close'] > prev_high) | downtrend
        exit_short = (data['close'] < prev_low) | uptrend
        
        signals.loc[buy_condition] = 'BUY'
        signals.loc[sell_condition] = 'SELL'
        signals.loc[exit_long] = 'CLOSE_LONG'
        signals.loc[exit_short] = 'CLOSE_SHORT'
        
        return signals


class RSIOverboughtOversoldStrategy(BaseStrategy):
    """
    RSI Overbought/Oversold Strategy
    
    More flexible exit based on RSI levels rather than moving averages
    """
    
    def __init__(self):
        super().__init__("RSI Overbought/Oversold")
        self.description = "Flexible RSI strategy with dynamic RSI-based exits"
        
        self.parameters = {
            'rsi_period': 2,
            'rsi_buy_threshold': 5,
            'rsi_sell_threshold': 95,
            'rsi_exit_long': 70,
            'rsi_exit_short': 30,
            'trend_ma_period': 50,  # Shorter MA for more responsive trend detection
            'stop_loss_pct': 0.01  # 1% stop loss
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI Overbought/Oversold signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Trend conditions
        uptrend = data['close'] > trend_ma
        downtrend = data['close'] < trend_ma
        
        # Entry conditions
        rsi_oversold = rsi_2 <= self.parameters['rsi_buy_threshold']
        rsi_overbought = rsi_2 >= self.parameters['rsi_sell_threshold']
        
        # Exit conditions (RSI-based)
        rsi_exit_long = rsi_2 >= self.parameters['rsi_exit_long']
        rsi_exit_short = rsi_2 <= self.parameters['rsi_exit_short']
        
        # Generate signals
        buy_condition = uptrend & rsi_oversold
        sell_condition = downtrend & rsi_overbought
        
        signals.loc[buy_condition] = 'BUY'
        signals.loc[sell_condition] = 'SELL'
        signals.loc[rsi_exit_long] = 'CLOSE_LONG'
        signals.loc[rsi_exit_short] = 'CLOSE_SHORT'
        
        return signals


class TPSStrategy(BaseStrategy):
    """
    TPS (Time Price Opportunity) Strategy
    
    Combines time, price, and momentum for entry timing
    """
    
    def __init__(self):
        super().__init__("TPS (Time Price Opportunity)")
        self.description = "Time-Price-Opportunity strategy combining multiple factors"
        
        self.parameters = {
            'rsi_period': 2,
            'rsi_threshold': 10,
            'ma_short': 5,
            'ma_long': 20,
            'days_since_high': 5,  # Days since recent high
            'volume_threshold': 1.2
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate TPS signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        ma_short = sma(data['close'], self.parameters['ma_short'])
        ma_long = sma(data['close'], self.parameters['ma_long'])
        vol_ratio = volume_ratio(data['volume'], 20)
        
        # Time component - days since high
        rolling_high = data['high'].rolling(20).max()
        days_since_high = pd.Series(0, index=data.index)
        
        for i in range(1, len(data)):
            if data['high'].iloc[i] >= rolling_high.iloc[i-1]:
                days_since_high.iloc[i] = 0
            else:
                days_since_high.iloc[i] = days_since_high.iloc[i-1] + 1
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # TPS conditions
        price_condition = data['close'] < ma_short  # Price below short MA
        time_condition = days_since_high >= self.parameters['days_since_high']
        opportunity_condition = (rsi_2 <= self.parameters['rsi_threshold']) & \
                               (vol_ratio >= self.parameters['volume_threshold'])
        trend_condition = ma_short > ma_long  # Uptrend
        
        # Exit condition
        exit_condition = data['close'] > ma_short
        
        # Generate signals
        buy_condition = trend_condition & price_condition & time_condition & opportunity_condition
        
        signals.loc[buy_condition] = 'BUY'
        signals.loc[exit_condition] = 'CLOSE_LONG'
        
        return signals


class CumulativeRSIStrategy(BaseStrategy):
    """
    Cumulative RSI Strategy
    
    Uses cumulative RSI readings over multiple days for stronger signals
    """
    
    def __init__(self):
        super().__init__("Cumulative RSI")
        self.description = "Strategy using cumulative RSI over multiple periods"
        
        self.parameters = {
            'rsi_period': 2,
            'cumulative_days': 3,
            'cumulative_threshold_buy': 15,  # Sum of RSI over 3 days < 15
            'cumulative_threshold_sell': 285,  # Sum of RSI over 3 days > 285
            'trend_ma_period': 200,
            'exit_ma_period': 5
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Cumulative RSI signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        exit_ma = sma(data['close'], self.parameters['exit_ma_period'])
        
        # Cumulative RSI
        cumulative_rsi = rsi_2.rolling(self.parameters['cumulative_days']).sum()
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Trend conditions
        uptrend = data['close'] > trend_ma
        downtrend = data['close'] < trend_ma
        
        # Cumulative conditions
        cum_oversold = cumulative_rsi <= self.parameters['cumulative_threshold_buy']
        cum_overbought = cumulative_rsi >= self.parameters['cumulative_threshold_sell']
        
        # Exit conditions
        exit_long = (data['close'] > exit_ma) | downtrend
        exit_short = (data['close'] < exit_ma) | uptrend
        
        # Generate signals
        buy_condition = uptrend & cum_oversold
        sell_condition = downtrend & cum_overbought
        
        signals.loc[buy_condition] = 'BUY'
        signals.loc[sell_condition] = 'SELL'
        signals.loc[exit_long] = 'CLOSE_LONG'
        signals.loc[exit_short] = 'CLOSE_SHORT'
        
        return signals


class RSI3Strategy(BaseStrategy):
    """
    Larry Connor's 3-Period RSI Strategy
    
    Variation of the famous 2-period RSI using 3-period RSI for slightly less aggressive entries
    """
    
    def __init__(self, variant: str = 'standard'):
        super().__init__(f"RSI-3 {variant.title()}")
        self.variant = variant
        self.description = f"3-Period RSI mean reversion strategy - {variant} variant"
        
        # Strategy parameters by variant
        if variant == 'standard':
            self.parameters = {
                'rsi_period': 3,
                'rsi_buy_threshold': 5,
                'rsi_sell_threshold': 95,
                'trend_ma_period': 200,
                'exit_ma_period': 5,
                'volume_threshold': 1.0
            }
        elif variant == 'conservative':
            self.parameters = {
                'rsi_period': 3,
                'rsi_buy_threshold': 3,
                'rsi_sell_threshold': 97,
                'trend_ma_period': 200,
                'exit_ma_period': 5,
                'volume_threshold': 1.5
            }
        elif variant == 'aggressive':
            self.parameters = {
                'rsi_period': 3,
                'rsi_buy_threshold': 10,
                'rsi_sell_threshold': 90,
                'trend_ma_period': 100,
                'exit_ma_period': 3,
                'volume_threshold': 0.8
            }
        elif variant == 'no_volume':
            self.parameters = {
                'rsi_period': 3,
                'rsi_buy_threshold': 5,
                'rsi_sell_threshold': 95,
                'trend_ma_period': 200,
                'exit_ma_period': 5,
                'volume_threshold': 0.0  # No volume filter
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI-3 signals"""
        # Calculate indicators
        rsi_3 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        exit_ma = sma(data['close'], self.parameters['exit_ma_period'])
        
        # Volume filter (if applicable)
        if self.parameters['volume_threshold'] > 0:
            vol_ratio = volume_ratio(data['volume'], 20)
            volume_ok = vol_ratio >= self.parameters['volume_threshold']
        else:
            volume_ok = pd.Series(True, index=data.index)
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Entry conditions
        uptrend = data['close'] > trend_ma
        downtrend = data['close'] < trend_ma
        
        # RSI conditions
        rsi_oversold = rsi_3 <= self.parameters['rsi_buy_threshold']
        rsi_overbought = rsi_3 >= self.parameters['rsi_sell_threshold']
        
        # Exit conditions
        above_exit_ma = data['close'] > exit_ma
        below_exit_ma = data['close'] < exit_ma
        
        # Generate signals
        buy_condition = uptrend & rsi_oversold & volume_ok
        sell_condition = downtrend & rsi_overbought & volume_ok
        
        # Exit conditions - restore original logic
        exit_long = above_exit_ma | downtrend
        exit_short = below_exit_ma | uptrend
        
        # Apply signals with proper priority
        signals.loc[buy_condition] = 'BUY'
        signals.loc[sell_condition] = 'SELL'
        
        # Apply exit signals only where we don't have entry signals
        signals.loc[exit_long & ~buy_condition & ~sell_condition] = 'CLOSE_LONG'
        signals.loc[exit_short & ~buy_condition & ~sell_condition] = 'CLOSE_SHORT'
        
        return signals


class HighProbabilityStrategy(BaseStrategy):
    """
    High Probability Strategy
    
    Combines multiple Larry Connor concepts for higher probability trades
    """
    
    def __init__(self):
        super().__init__("High Probability Combo")
        self.description = "High probability strategy combining multiple Larry Connor concepts"
        
        self.parameters = {
            'rsi_period': 2,
            'rsi_threshold': 5,
            'trend_ma_period': 200,
            'intermediate_ma_period': 50,
            'short_ma_period': 10,
            'volume_threshold': 1.5,
            'atr_period': 14,
            'days_down': 2  # Consecutive down days
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate High Probability signals"""
        # Calculate indicators
        rsi_2 = rsi(data['close'], self.parameters['rsi_period'])
        trend_ma = sma(data['close'], self.parameters['trend_ma_period'])
        intermediate_ma = sma(data['close'], self.parameters['intermediate_ma_period'])
        short_ma = sma(data['close'], self.parameters['short_ma_period'])
        vol_ratio = volume_ratio(data['volume'], 20)
        atr_14 = atr(data['high'], data['low'], data['close'], self.parameters['atr_period'])
        
        # Consecutive down days
        down_days = (data['close'] < data['close'].shift(1)).astype(int)
        consecutive_down = down_days.rolling(self.parameters['days_down']).sum()
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Multiple confirmation conditions
        strong_uptrend = (data['close'] > trend_ma) & (data['close'] > intermediate_ma)
        pullback_condition = data['close'] < short_ma
        rsi_oversold = rsi_2 <= self.parameters['rsi_threshold']
        volume_confirmation = vol_ratio >= self.parameters['volume_threshold']
        time_confirmation = consecutive_down >= self.parameters['days_down']
        
        # Exit conditions
        exit_condition = (data['close'] > short_ma) | (data['close'] < trend_ma)
        
        # Generate signals (only long positions for this high-probability approach)
        buy_condition = (strong_uptrend & pullback_condition & rsi_oversold & 
                        volume_confirmation & time_confirmation)
        
        signals.loc[buy_condition] = 'BUY'
        signals.loc[exit_condition] = 'CLOSE_LONG'
        
        return signals


# Strategy Registry
LARRY_CONNOR_STRATEGIES = {
    # RSI-2 Variants
    'rsi2_standard': lambda: RSI2Strategy('standard'),
    'rsi2_conservative': lambda: RSI2Strategy('conservative'),
    'rsi2_aggressive': lambda: RSI2Strategy('aggressive'),
    'rsi2_no_volume': lambda: RSI2Strategy('no_volume'),
    
    # RSI-3 Variants
    'rsi3_standard': lambda: RSI3Strategy('standard'),
    'rsi3_conservative': lambda: RSI3Strategy('conservative'),
    'rsi3_aggressive': lambda: RSI3Strategy('aggressive'),
    'rsi3_no_volume': lambda: RSI3Strategy('no_volume'),
    
    # Double 7's Variants
    'double7_standard': lambda: Double7Strategy('standard'),
    'double7_conservative': lambda: Double7Strategy('conservative'),
    'double7_aggressive': lambda: Double7Strategy('aggressive'),
    'double10': lambda: Double7Strategy('double_10'),
    
    # Additional Strategies
    'rsi_pullback_3': lambda: RSIPullbackStrategy(3),
    'rsi_pullback_4': lambda: RSIPullbackStrategy(4),
    'rsi_overbought_oversold': lambda: RSIOverboughtOversoldStrategy(),
    'tps': lambda: TPSStrategy(),
    'cumulative_rsi': lambda: CumulativeRSIStrategy(),
    'high_probability': lambda: HighProbabilityStrategy(),
}


def get_strategy(strategy_name: str) -> BaseStrategy:
    """Get a strategy by name"""
    if strategy_name not in LARRY_CONNOR_STRATEGIES:
        available = list(LARRY_CONNOR_STRATEGIES.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
    
    return LARRY_CONNOR_STRATEGIES[strategy_name]()


def list_strategies() -> List[str]:
    """List all available Larry Connor strategies"""
    return list(LARRY_CONNOR_STRATEGIES.keys())


def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
    """Get information about a specific strategy"""
    strategy = get_strategy(strategy_name)
    return strategy.get_info()


class StrategyComparison:
    """
    Compare multiple Larry Connor strategies
    """
    
    def __init__(self):
        self.results = {}
    
    def add_strategy_result(self, strategy_name: str, backtest_results: Dict[str, Any]):
        """Add backtest results for a strategy"""
        self.results[strategy_name] = backtest_results
    
    def compare_performance(self) -> pd.DataFrame:
        """Compare performance metrics across strategies"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = {}
        
        for strategy_name, results in self.results.items():
            metrics = results.get('metrics', {})
            
            comparison_data[strategy_name] = {
                'Total Return': metrics.get('total_return', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Win Rate': metrics.get('win_rate', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Profit Factor': metrics.get('profit_factor', 0),
                'Annual Return': metrics.get('annual_return', 0),
                'Volatility': metrics.get('volatility', 0)
            }
        
        df = pd.DataFrame(comparison_data).T
        
        # Sort by Sharpe ratio (descending)
        if 'Sharpe Ratio' in df.columns:
            df = df.sort_values('Sharpe Ratio', ascending=False)
        
        return df
    
    def rank_strategies(self) -> pd.DataFrame:
        """Rank strategies using a composite score"""
        comparison_df = self.compare_performance()
        
        if comparison_df.empty:
            return pd.DataFrame()
        
        # Normalize metrics (0-1 scale)
        normalized_df = comparison_df.copy()
        
        # Higher is better metrics
        positive_metrics = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Profit Factor', 'Annual Return']
        for metric in positive_metrics:
            if metric in normalized_df.columns:
                col = normalized_df[metric]
                if col.max() != col.min():
                    normalized_df[metric] = (col - col.min()) / (col.max() - col.min())
                else:
                    normalized_df[metric] = 0.5
        
        # Lower is better metrics
        negative_metrics = ['Max Drawdown', 'Volatility']
        for metric in negative_metrics:
            if metric in normalized_df.columns:
                col = normalized_df[metric]
                if col.max() != col.min():
                    normalized_df[metric] = 1 - (col - col.min()) / (col.max() - col.min())
                else:
                    normalized_df[metric] = 0.5
        
        # Calculate composite score
        weights = {
            'Total Return': 0.25,
            'Sharpe Ratio': 0.25,
            'Max Drawdown': 0.20,
            'Win Rate': 0.15,
            'Profit Factor': 0.10,
            'Volatility': 0.05
        }
        
        composite_score = pd.Series(0.0, index=normalized_df.index)
        for metric, weight in weights.items():
            if metric in normalized_df.columns:
                composite_score += normalized_df[metric] * weight
        
        # Add composite score to original dataframe
        result_df = comparison_df.copy()
        result_df['Composite Score'] = composite_score
        result_df = result_df.sort_values('Composite Score', ascending=False)
        
        return result_df
    
    def generate_report(self) -> str:
        """Generate a comprehensive comparison report"""
        if not self.results:
            return "No strategy results to compare."
        
        ranking_df = self.rank_strategies()
        
        report = f"""
📊 LARRY CONNOR STRATEGIES COMPARISON REPORT
{'=' * 60}

📈 STRATEGY RANKINGS (by Composite Score):
{'-' * 40}
"""
        
        for i, (strategy, row) in enumerate(ranking_df.iterrows(), 1):
            report += f"""
{i}. {strategy.upper()}
   • Total Return: {row['Total Return']:.2%}
   • Sharpe Ratio: {row['Sharpe Ratio']:.2f}
   • Max Drawdown: {row['Max Drawdown']:.2%}
   • Win Rate: {row['Win Rate']:.1%}
   • Total Trades: {row['Total Trades']:.0f}
   • Composite Score: {row['Composite Score']:.3f}
"""
        
        # Best performer analysis
        best_strategy = ranking_df.index[0]
        best_metrics = ranking_df.iloc[0]
        
        report += f"""

🏆 BEST OVERALL PERFORMER: {best_strategy.upper()}
{'-' * 40}
This strategy achieved the highest composite score of {best_metrics['Composite Score']:.3f}
with a {best_metrics['Total Return']:.2%} return and {best_metrics['Sharpe Ratio']:.2f} Sharpe ratio.

📊 KEY INSIGHTS:
• Highest Return: {ranking_df['Total Return'].idxmax()} ({ranking_df['Total Return'].max():.2%})
• Best Sharpe Ratio: {ranking_df['Sharpe Ratio'].idxmax()} ({ranking_df['Sharpe Ratio'].max():.2f})
• Lowest Drawdown: {ranking_df['Max Drawdown'].idxmin()} ({ranking_df['Max Drawdown'].min():.2%})
• Highest Win Rate: {ranking_df['Win Rate'].idxmax()} ({ranking_df['Win Rate'].max():.1%})

💡 RECOMMENDATIONS:
1. Consider {best_strategy} for optimal risk-adjusted returns
2. For conservative approach, focus on strategies with low drawdown
3. For active trading, consider strategies with higher trade frequency
4. Always validate results with out-of-sample testing
"""
        
        return report 