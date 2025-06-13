"""
Performance Analysis Module
Implementation of Phase 3 - Advanced Analytics & Optimization

This module provides comprehensive performance analysis tools for
strategy evaluation, comparison, and optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import risk metrics for calculations
from ..risk_management.risk_metrics import RiskMetricsCalculator


class PerformanceAnalyzer:
    """
    Comprehensive Performance Analysis for Trading Strategies
    
    Analyzes strategy performance with detailed metrics, visualizations,
    and comparative analysis capabilities.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize Performance Analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.risk_calculator = RiskMetricsCalculator(risk_free_rate)
        
    def analyze_strategy(self, 
                        returns: pd.Series,
                        trades: pd.DataFrame = None,
                        benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        Comprehensive strategy analysis
        
        Args:
            returns: Strategy returns series
            trades: DataFrame with trade details
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        # Basic performance metrics
        basic_metrics = self.risk_calculator.calculate_all_metrics(returns)
        
        # Trade analysis
        trade_metrics = self._analyze_trades(trades) if trades is not None else {}
        
        # Benchmark comparison
        benchmark_metrics = self._compare_to_benchmark(returns, benchmark_returns) if benchmark_returns is not None else {}
        
        # Period analysis
        period_metrics = self._analyze_periods(returns)
        
        return {
            'basic_metrics': basic_metrics,
            'trade_metrics': trade_metrics,
            'benchmark_metrics': benchmark_metrics,
            'period_metrics': period_metrics,
            'analysis_date': datetime.now()
        }
    
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade-level performance"""
        if trades.empty:
            return {}
        
        # Calculate trade P&L
        if 'pnl' in trades.columns:
            trade_pnl = trades['pnl']
        else:
            return {}
        
        winning_trades = trade_pnl[trade_pnl > 0]
        losing_trades = trade_pnl[trade_pnl < 0]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if len(trades) > 0 else 0,
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else np.inf,
            'total_pnl': trade_pnl.sum()
        }
    
    def _compare_to_benchmark(self, returns: pd.Series, benchmark: pd.Series) -> Dict[str, Any]:
        """Compare strategy to benchmark"""
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        # Calculate excess returns
        excess_returns = aligned_returns - aligned_benchmark
        
        # Calculate metrics
        strategy_total = (1 + aligned_returns).prod() - 1
        benchmark_total = (1 + aligned_benchmark).prod() - 1
        
        # Beta calculation
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
        
        return {
            'total_return_strategy': strategy_total,
            'total_return_benchmark': benchmark_total,
            'excess_return': strategy_total - benchmark_total,
            'beta': beta,
            'correlation': aligned_returns.corr(aligned_benchmark),
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        }
    
    def _analyze_periods(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance by different time periods"""
        if len(returns) == 0:
            return {}
        
        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'monthly_returns': {
                'mean': monthly_returns.mean(),
                'std': monthly_returns.std(),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': (monthly_returns > 0).sum()
            }
        }
    
    def create_performance_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a formatted performance report"""
        report = []
        report.append("=" * 60)
        report.append("STRATEGY PERFORMANCE ANALYSIS")
        report.append("=" * 60)
        
        # Basic Metrics
        basic = analysis_results.get('basic_metrics', {})
        if basic:
            report.append("\nBASIC METRICS:")
            report.append(f"Total Return:        {basic.get('total_return', 0):.2%}")
            report.append(f"Sharpe Ratio:        {basic.get('sharpe_ratio', 0):.3f}")
            report.append(f"Max Drawdown:        {basic.get('max_drawdown', 0):.2%}")
            report.append(f"Win Rate:            {basic.get('win_rate', 0):.2%}")
        
        # Trade Metrics
        trades = analysis_results.get('trade_metrics', {})
        if trades:
            report.append("\nTRADE ANALYSIS:")
            report.append(f"Total Trades:        {trades.get('total_trades', 0)}")
            report.append(f"Win Rate:            {trades.get('win_rate', 0):.2%}")
            report.append(f"Profit Factor:       {trades.get('profit_factor', 0):.2f}")
        
        return "\n".join(report)


class StrategyComparison:
    """
    Compare multiple strategies across various metrics
    """
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
    
    def compare_strategies(self, 
                          strategy_returns: Dict[str, pd.Series],
                          benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategy_returns: Dictionary mapping strategy names to return series
            benchmark_returns: Optional benchmark for comparison
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_results = {}
        
        for name, returns in strategy_returns.items():
            analysis = self.analyzer.analyze_strategy(
                returns, 
                benchmark_returns=benchmark_returns
            )
            
            # Extract key metrics for comparison
            basic = analysis.get('basic_metrics', {})
            metrics = {
                'Total Return': basic.get('total_return', 0),
                'Sharpe Ratio': basic.get('sharpe_ratio', 0),
                'Max Drawdown': basic.get('max_drawdown', 0),
                'Volatility': basic.get('volatility', 0),
                'Win Rate': basic.get('win_rate', 0)
            }
            
            # Benchmark metrics if available
            benchmark = analysis.get('benchmark_metrics', {})
            if benchmark:
                metrics.update({
                    'Beta': benchmark.get('beta', 0),
                    'Excess Return': benchmark.get('excess_return', 0)
                })
            
            comparison_results[name] = metrics
        
        return pd.DataFrame(comparison_results).T
    
    def rank_strategies(self, 
                       comparison_df: pd.DataFrame,
                       ranking_metric: str = 'Sharpe Ratio') -> pd.DataFrame:
        """Rank strategies by specified metric"""
        if ranking_metric not in comparison_df.columns:
            raise ValueError(f"Metric '{ranking_metric}' not found")
        
        ranked = comparison_df.sort_values(ranking_metric, ascending=False)
        ranked['Rank'] = range(1, len(ranked) + 1)
        
        return ranked


class RollingPerformanceAnalyzer:
    """
    Analyze rolling performance metrics over time
    """
    
    def __init__(self, window: int = 252):
        """
        Initialize Rolling Performance Analyzer
        
        Args:
            window: Rolling window size in periods
        """
        self.window = window
        self.analyzer = PerformanceAnalyzer()
    
    def calculate_rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_sharpe = []
        
        for i in range(self.window, len(returns) + 1):
            window_returns = returns.iloc[i-self.window:i]
            excess_returns = window_returns - 0.02/252  # Risk-free rate
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
            rolling_sharpe.append(sharpe)
        
        # Ensure the index length matches the data length
        if len(rolling_sharpe) > 0:
            rolling_index = returns.index[self.window-1:self.window-1+len(rolling_sharpe)]
            return pd.Series(rolling_sharpe, index=rolling_index)
        else:
            return pd.Series([], dtype=float)


class DrawdownAnalyzer:
    """
    Detailed drawdown analysis and visualization
    """
    
    def analyze_drawdowns(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with drawdown analysis
        """
        if len(returns) == 0:
            return {}
        
        # Calculate cumulative returns and drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Calculate statistics
        max_drawdown = drawdown.min()
        in_drawdown = drawdown < 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': drawdown[in_drawdown].mean() if in_drawdown.any() else 0,
            'time_in_drawdown_pct': in_drawdown.sum() / len(drawdown) * 100,
            'drawdown_series': drawdown,
            'cumulative_returns': cumulative
        } 