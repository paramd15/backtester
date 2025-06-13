"""
Trend Following Strategies
Implementation of Phase 1.3 from roadmap - Trend Following Strategies

This module implements advanced trend-following strategies that capture
sustained price movements in trending markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_strategy import BaseStrategy
from ..indicators.basic_trend import sma, ema, macd, macd_signal, macd_histogram
from ..indicators.momentum import rsi
from ..indicators.volatility import atr
from ..indicators.volume import volume_ratio
from ..indicators.advanced_trend import parabolic_sar, adx, triple_moving_average


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Classic trend following strategy using two moving averages
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20, 
                 ma_type: str = 'sma', variant: str = 'standard'):
        super().__init__(f"MA Crossover {fast_period}/{slow_period} {ma_type.upper()} {variant.title()}")
        self.description = f"{ma_type.upper()} crossover strategy ({fast_period}/{slow_period}) - {variant}"
        
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'ma_type': ma_type,
            'volume_filter': variant != 'simple',
            'trend_filter': variant == 'filtered',
            'rsi_filter': variant == 'rsi_filtered',
            'atr_filter': variant == 'volatility_filtered'
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate moving average crossover signals"""
        # Calculate moving averages
        if self.parameters['ma_type'] == 'sma':
            fast_ma = sma(data['close'], self.parameters['fast_period'])
            slow_ma = sma(data['close'], self.parameters['slow_period'])
        else:  # ema
            fast_ma = ema(data['close'], self.parameters['fast_period'])
            slow_ma = ema(data['close'], self.parameters['slow_period'])
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Basic crossover conditions
        bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Optional filters
        filter_condition = pd.Series(True, index=data.index)
        
        if self.parameters['volume_filter']:
            vol_ratio = volume_ratio(data['volume'], 20)
            filter_condition &= vol_ratio > 1.0
        
        if self.parameters['trend_filter']:
            long_ma = sma(data['close'], 50)
            filter_condition &= data['close'] > long_ma
        
        if self.parameters['rsi_filter']:
            rsi_14 = rsi(data['close'], 14)
            filter_condition &= (rsi_14 > 30) & (rsi_14 < 70)
        
        if self.parameters['atr_filter']:
            current_atr = atr(data['high'], data['low'], data['close'], 14)
            avg_atr = current_atr.rolling(20).mean()
            filter_condition &= current_atr > avg_atr
        
        # Generate signals
        signals.loc[bullish_cross & filter_condition] = 'BUY'
        signals.loc[bearish_cross & filter_condition] = 'SELL'
        
        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy
    
    Trend following strategy using MACD indicator
    """
    
    def __init__(self, variant: str = 'standard'):
        super().__init__(f"MACD {variant.title()}")
        self.description = f"MACD trend following strategy - {variant} variant"
        
        if variant == 'standard':
            self.parameters = {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'entry_method': 'signal_cross',  # 'signal_cross' or 'zero_cross'
                'volume_filter': True,
                'trend_filter': False
            }
        elif variant == 'aggressive':
            self.parameters = {
                'fast_period': 8,
                'slow_period': 21,
                'signal_period': 5,
                'entry_method': 'signal_cross',
                'volume_filter': False,
                'trend_filter': False
            }
        elif variant == 'conservative':
            self.parameters = {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'entry_method': 'zero_cross',
                'volume_filter': True,
                'trend_filter': True
            }
        elif variant == 'histogram':
            self.parameters = {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'entry_method': 'histogram',
                'volume_filter': True,
                'trend_filter': False
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MACD signals"""
        # Calculate MACD
        macd_line = macd(data['close'], self.parameters['fast_period'], self.parameters['slow_period'])
        signal_line = macd_signal(data['close'], self.parameters['fast_period'], 
                                 self.parameters['slow_period'], self.parameters['signal_period'])
        histogram = macd_histogram(data['close'], self.parameters['fast_period'], 
                                  self.parameters['slow_period'], self.parameters['signal_period'])
        
        # Initialize signals
        signals = pd.Series('HOLD', index=data.index)
        
        # Entry conditions based on method
        if self.parameters['entry_method'] == 'signal_cross':
            # MACD line crosses above/below signal line
            bullish_signal = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            bearish_signal = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        elif self.parameters['entry_method'] == 'zero_cross':
            # MACD line crosses above/below zero
            bullish_signal = (macd_line > 0) & (macd_line.shift(1) <= 0)
            bearish_signal = (macd_line < 0) & (macd_line.shift(1) >= 0)
        
        elif self.parameters['entry_method'] == 'histogram':
            # Histogram changes direction
            bullish_signal = (histogram > 0) & (histogram.shift(1) <= 0)
            bearish_signal = (histogram < 0) & (histogram.shift(1) >= 0)
        
        # Optional filters
        filter_condition = pd.Series(True, index=data.index)
        
        if self.parameters['volume_filter']:
            vol_ratio = volume_ratio(data['volume'], 20)
            filter_condition &= vol_ratio > 1.0
        
        if self.parameters['trend_filter']:
            trend_ma = sma(data['close'], 50)
            filter_condition &= data['close'] > trend_ma
        
        # Generate signals
        signals.loc[bullish_signal & filter_condition] = 'BUY'
        signals.loc[bearish_signal & filter_condition] = 'SELL'
        
        return signals


class ParabolicSARTrendStrategy(BaseStrategy):
    """Parabolic SAR Trend Following Strategy"""
    
    def __init__(self,
                 af_start: float = 0.02,
                 af_increment: float = 0.02,
                 af_max: float = 0.2,
                 trend_filter: Optional[str] = None,
                 position_size: float = 0.1):
        parameters = {
            'af_start': af_start,
            'af_increment': af_increment,
            'af_max': af_max,
            'trend_filter': trend_filter,
            'position_size': position_size
        }
        
        super().__init__("Parabolic SAR Trend", parameters)
        
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        self.trend_filter = trend_filter
        self.position_size = position_size
        
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """Generate trading signals - required by BaseStrategy"""
        pass
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Parabolic SAR trend signals"""
        df = data.copy()
        
        # Calculate Parabolic SAR
        df['PSAR'] = parabolic_sar(df['High'], df['Low'], 
                                  self.af_start, self.af_increment, self.af_max)
        
        # Initialize signals
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        # Generate signals
        for i in range(1, len(df)):
            # Buy when price crosses above PSAR
            if df.iloc[i]['Close'] > df.iloc[i]['PSAR'] and df.iloc[i-1]['Close'] <= df.iloc[i-1]['PSAR']:
                df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.8
                df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
                
            # Sell when price crosses below PSAR
            elif df.iloc[i]['Close'] < df.iloc[i]['PSAR'] and df.iloc[i-1]['Close'] >= df.iloc[i-1]['PSAR']:
                df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.8
                df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
        
        return df


class ADXTrendStrengthStrategy(BaseStrategy):
    """ADX Trend Strength Strategy"""
    
    def __init__(self,
                 adx_period: int = 14,
                 adx_threshold: float = 25,
                 di_crossover: bool = True,
                 position_size: float = 0.12):
        parameters = {
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'di_crossover': di_crossover,
            'position_size': position_size
        }
        
        super().__init__("ADX Trend Strength", parameters)
        
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.di_crossover = di_crossover
        self.position_size = position_size
        
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """Generate trading signals - required by BaseStrategy"""
        pass
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ADX trend strength signals"""
        df = data.copy()
        
        # Calculate ADX and Directional Indicators
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = adx(df['High'], df['Low'], df['Close'], self.adx_period)
        
        # Initialize signals
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        # Generate signals
        for i in range(self.adx_period, len(df)):
            adx_val = df.iloc[i]['ADX']
            plus_di = df.iloc[i]['Plus_DI']
            minus_di = df.iloc[i]['Minus_DI']
            
            # Check for strong trend
            if adx_val > self.adx_threshold:
                if self.di_crossover:
                    # Check for DI crossover
                    prev_plus_di = df.iloc[i-1]['Plus_DI']
                    prev_minus_di = df.iloc[i-1]['Minus_DI']
                    
                    # Bullish crossover
                    if plus_di > minus_di and prev_plus_di <= prev_minus_di:
                        signal_strength = min(adx_val / 50, 1.0)
                        df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                        df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
                        df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
                    
                    # Bearish crossover
                    elif minus_di > plus_di and prev_minus_di <= prev_plus_di:
                        signal_strength = min(adx_val / 50, 1.0)
                        df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                        df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
                        df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
        
        return df


class TripleMovingAverageStrategy(BaseStrategy):
    """Triple Moving Average Strategy"""
    
    def __init__(self,
                 ma1_period: int = 5,
                 ma2_period: int = 13,
                 ma3_period: int = 21,
                 ma_type: str = 'sma',
                 position_size: float = 0.15):
        parameters = {
            'ma1_period': ma1_period,
            'ma2_period': ma2_period,
            'ma3_period': ma3_period,
            'ma_type': ma_type,
            'position_size': position_size
        }
        
        super().__init__("Triple Moving Average", parameters)
        
        self.ma1_period = ma1_period
        self.ma2_period = ma2_period
        self.ma3_period = ma3_period
        self.ma_type = ma_type
        self.position_size = position_size
        
    def generate_signals(self, backtester, daily_data: pd.DataFrame) -> None:
        """Generate trading signals - required by BaseStrategy"""
        pass
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Triple MA signals"""
        df = data.copy()
        
        # Calculate Triple Moving Averages
        df['MA1'], df['MA2'], df['MA3'], df['MA_Signal'] = triple_moving_average(
            df['Close'], self.ma1_period, self.ma2_period, self.ma3_period, self.ma_type)
        
        # Initialize signals
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        # Generate signals
        for i in range(max(self.ma1_period, self.ma2_period, self.ma3_period), len(df)):
            ma_signal = df.iloc[i]['MA_Signal']
            prev_ma_signal = df.iloc[i-1]['MA_Signal']
            
            # Signal on MA alignment change
            if ma_signal == 1 and prev_ma_signal != 1:
                df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.7
                df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
                
            elif ma_signal == -1 and prev_ma_signal != -1:
                df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                df.iloc[i, df.columns.get_loc('Signal_Strength')] = 0.7
                df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i]['Close']
        
        return df


# Strategy registry
TREND_FOLLOWING_STRATEGIES = {
    'parabolic_sar_trend': ParabolicSARTrendStrategy,
    'adx_trend_strength': ADXTrendStrengthStrategy,
    'triple_ma_system': TripleMovingAverageStrategy,
}


def get_strategy(strategy_name: str, **kwargs):
    """Factory function to create trend following strategies"""
    if strategy_name not in TREND_FOLLOWING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = TREND_FOLLOWING_STRATEGIES[strategy_name]
    return strategy_class(**kwargs) 