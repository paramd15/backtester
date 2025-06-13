"""
Performance Metrics and Risk Analysis
Additional metrics and analysis tools to complement QuantStats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class RiskMetrics:
    """Risk metrics container"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    volatility: float
    downside_deviation: float
    ulcer_index: float


class PerformanceMetricsCalculator:
    """
    Additional performance analysis tools
    Complements QuantStats with custom metrics and analysis
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_custom_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate custom performance metrics not covered by QuantStats
        
        Args:
            backtest_results: Results from backtester
            
        Returns:
            Dictionary of custom metrics
        """
        if 'portfolio_history' not in backtest_results:
            return {}
        
        portfolio_history = backtest_results['portfolio_history']
        returns = portfolio_history['total_value'].pct_change().dropna()
        
        metrics = {}
        
        # Trading-specific metrics
        if 'signals_history' in backtest_results:
            signals = backtest_results['signals_history']
            metrics.update(self._calculate_trading_metrics(signals, portfolio_history))
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        metrics.update(risk_metrics.__dict__)
        
        # Performance consistency metrics
        metrics.update(self._calculate_consistency_metrics(returns))
        
        # Market timing metrics
        metrics.update(self._calculate_timing_metrics(returns))
        
        return metrics
    
    def _calculate_trading_metrics(self, signals: pd.DataFrame, 
                                 portfolio_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        metrics = {}
        
        if signals.empty:
            return metrics
        
        # Trade frequency
        total_trades = len(signals[signals['signal'].isin(['BUY', 'SELL'])])
        trading_days = len(portfolio_history)
        
        metrics['trades_per_month'] = (total_trades / trading_days) * 21 if trading_days > 0 else 0
        metrics['total_trades'] = total_trades
        
        # Signal distribution
        buy_signals = len(signals[signals['signal'] == 'BUY'])
        sell_signals = len(signals[signals['signal'] == 'SELL'])
        
        metrics['buy_sell_ratio'] = buy_signals / sell_signals if sell_signals > 0 else np.inf
        
        # Activity metrics
        active_days = len(signals[signals['signal'] != 'HOLD'])
        metrics['activity_ratio'] = active_days / trading_days if trading_days > 0 else 0
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration (days in drawdown)
        in_drawdown = drawdown < -0.01  # More than 1% drawdown
        drawdown_duration = in_drawdown.sum()
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Ulcer Index (alternative drawdown measure)
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration=drawdown_duration,
            volatility=volatility,
            downside_deviation=downside_deviation,
            ulcer_index=ulcer_index
        )
    
    def _calculate_consistency_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance consistency metrics"""
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # Ensure returns has a datetime index for resampling
        if not isinstance(returns.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            # If returns doesn't have datetime index, skip time-based resampling
            metrics['win_rate_daily'] = (returns > 0).sum() / len(returns)
            if returns.std() > 0:
                metrics['consistency_ratio'] = returns.mean() / returns.std()
            return metrics
        
        # Monthly consistency
        try:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if len(monthly_returns) > 1:
                positive_months = (monthly_returns > 0).sum()
                metrics['win_rate_monthly'] = positive_months / len(monthly_returns)
                metrics['monthly_volatility'] = monthly_returns.std()
        except Exception as e:
            pass  # Skip monthly metrics if resampling fails
        
        # Weekly consistency
        try:
            weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
            if len(weekly_returns) > 1:
                positive_weeks = (weekly_returns > 0).sum()
                metrics['win_rate_weekly'] = positive_weeks / len(weekly_returns)
        except Exception as e:
            pass  # Skip weekly metrics if resampling fails
        
        # Daily win rate
        positive_days = (returns > 0).sum()
        metrics['win_rate_daily'] = positive_days / len(returns)
        
        # Consistency ratio (average return / volatility)
        if returns.std() > 0:
            metrics['consistency_ratio'] = returns.mean() / returns.std()
        
        return metrics
    
    def _calculate_timing_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate market timing metrics"""
        if len(returns) < 20:
            return {}
        
        metrics = {}
        
        # Rolling performance windows
        rolling_20d = returns.rolling(20).sum()
        rolling_60d = returns.rolling(60).sum()
        
        # Timing consistency
        if len(rolling_20d.dropna()) > 0:
            metrics['rolling_20d_positive_ratio'] = (rolling_20d > 0).sum() / len(rolling_20d.dropna())
        
        if len(rolling_60d.dropna()) > 0:
            metrics['rolling_60d_positive_ratio'] = (rolling_60d > 0).sum() / len(rolling_60d.dropna())
        
        return metrics
    
    def generate_performance_summary(self, backtest_results: Dict[str, Any],
                                   strategy_name: str = "Strategy") -> str:
        """
        Generate a comprehensive performance summary report
        
        Args:
            backtest_results: Results from backtester
            strategy_name: Name of the strategy
            
        Returns:
            Formatted performance summary string
        """
        metrics = self.calculate_custom_metrics(backtest_results)
        
        if not metrics:
            return f"No performance data available for {strategy_name}"
        
        summary = []
        summary.append("=" * 60)
        summary.append(f"PERFORMANCE SUMMARY: {strategy_name}")
        summary.append("=" * 60)
        
        # Basic metrics
        if 'total_trades' in metrics:
            summary.append(f"\nTRADING ACTIVITY:")
            summary.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
            summary.append(f"  Trades per Month: {metrics.get('trades_per_month', 0):.1f}")
            summary.append(f"  Activity Ratio: {metrics.get('activity_ratio', 0):.2%}")
        
        # Risk metrics
        summary.append(f"\nRISK METRICS:")
        summary.append(f"  VaR (95%): {metrics.get('var_95', 0):.2%}")
        summary.append(f"  CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
        summary.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        summary.append(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        summary.append(f"  Downside Deviation: {metrics.get('downside_deviation', 0):.2%}")
        
        # Consistency metrics
        summary.append(f"\nCONSISTENCY METRICS:")
        summary.append(f"  Daily Win Rate: {metrics.get('win_rate_daily', 0):.2%}")
        if 'win_rate_weekly' in metrics:
            summary.append(f"  Weekly Win Rate: {metrics.get('win_rate_weekly', 0):.2%}")
        if 'win_rate_monthly' in metrics:
            summary.append(f"  Monthly Win Rate: {metrics.get('win_rate_monthly', 0):.2%}")
        
        return "\n".join(summary)
    
    def compare_risk_profiles(self, strategy_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare risk profiles of multiple strategies
        
        Args:
            strategy_results: Dictionary mapping strategy names to backtest results
            
        Returns:
            DataFrame with risk comparison metrics
        """
        comparison_data = {}
        
        for strategy_name, results in strategy_results.items():
            metrics = self.calculate_custom_metrics(results)
            
            risk_profile = {
                'VaR_95': metrics.get('var_95', 0),
                'CVaR_95': metrics.get('cvar_95', 0),
                'Max_Drawdown': metrics.get('max_drawdown', 0),
                'Volatility': metrics.get('volatility', 0),
                'Downside_Deviation': metrics.get('downside_deviation', 0),
                'Ulcer_Index': metrics.get('ulcer_index', 0),
                'Win_Rate_Daily': metrics.get('win_rate_daily', 0),
                'Consistency_Ratio': metrics.get('consistency_ratio', 0)
            }
            
            comparison_data[strategy_name] = risk_profile
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Sort by a composite risk score (lower is better)
        if len(comparison_df) > 0:
            comparison_df['Risk_Score'] = (
                abs(comparison_df['Max_Drawdown']) * 0.3 +
                comparison_df['Volatility'] * 0.3 +
                abs(comparison_df['VaR_95']) * 0.2 +
                comparison_df['Downside_Deviation'] * 0.2
            )
            comparison_df = comparison_df.sort_values('Risk_Score')
        
        return comparison_df


# Backward compatibility - keep the original class name
class PerformanceAnalyzer(PerformanceMetricsCalculator):
    """Alias for backward compatibility"""
    pass 