"""
QuantStats Integration
Professional performance analysis using QuantStats library
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import warnings

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    warnings.warn("QuantStats not available. Install with: pip install quantstats")

import matplotlib.pyplot as plt
from datetime import datetime


class QuantStatsAnalyzer:
    """
    Professional performance analysis using QuantStats
    
    Provides institutional-grade performance reporting including:
    - Tearsheet generation
    - Risk-adjusted metrics
    - Benchmark comparisons
    - Drawdown analysis
    - Rolling performance windows
    """
    
    def __init__(self, benchmark_symbol: str = 'SPY'):
        """
        Initialize QuantStats analyzer
        
        Args:
            benchmark_symbol: Benchmark symbol for comparison (default: SPY)
        """
        if not QUANTSTATS_AVAILABLE:
            raise ImportError("QuantStats is required. Install with: pip install quantstats")
        
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_data = None
        
        # Configure QuantStats
        qs.extend_pandas()
    
    def prepare_returns_series(self, backtest_results: Dict[str, Any]) -> pd.Series:
        """
        Convert backtest results to returns series for QuantStats
        
        Args:
            backtest_results: Results from backtester
            
        Returns:
            Daily returns series
        """
        if 'portfolio_history' not in backtest_results:
            raise ValueError("Portfolio history not found in backtest results")
        
        portfolio_history = backtest_results['portfolio_history']
        
        if portfolio_history.empty:
            raise ValueError("Empty portfolio history")
        
        # Calculate daily returns from portfolio values
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        # Ensure proper datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        return returns
    
    def load_benchmark_data(self, benchmark_returns: Optional[pd.Series] = None) -> pd.Series:
        """
        Load benchmark data for comparison
        
        Args:
            benchmark_returns: Pre-calculated benchmark returns (optional)
            
        Returns:
            Benchmark returns series
        """
        if benchmark_returns is not None:
            self.benchmark_data = benchmark_returns
            return benchmark_returns
        
        # For now, create a simple benchmark (market return simulation)
        # In production, you'd load actual SPY/benchmark data
        print(f"⚠️  Using simulated benchmark data. In production, load actual {self.benchmark_symbol} data.")
        
        # Create simple benchmark returns (random walk with slight positive drift)
        np.random.seed(42)  # For reproducibility
        n_days = 500  # Approximate 2 years
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.01, n_days),  # ~12% annual return, 16% volatility
            index=pd.date_range(start='2023-01-01', periods=n_days, freq='D'),
            name=self.benchmark_symbol
        )
        
        self.benchmark_data = benchmark_returns
        return benchmark_returns
    
    def generate_full_tearsheet(self, backtest_results: Dict[str, Any], 
                               strategy_name: str = "Strategy",
                               output_file: Optional[str] = None,
                               benchmark_returns: Optional[pd.Series] = None) -> None:
        """
        Generate comprehensive QuantStats tearsheet
        
        Args:
            backtest_results: Results from backtester
            strategy_name: Name of the strategy
            output_file: Path to save HTML report (optional)
            benchmark_returns: Benchmark returns for comparison
        """
        # Prepare returns
        returns = self.prepare_returns_series(backtest_results)
        benchmark = self.load_benchmark_data(benchmark_returns)
        
        # Align benchmark with strategy returns
        common_dates = returns.index.intersection(benchmark.index)
        if len(common_dates) == 0:
            print("⚠️  No common dates between strategy and benchmark. Using strategy period for benchmark.")
            # Create benchmark for strategy period
            benchmark = pd.Series(
                np.random.normal(0.0005, 0.01, len(returns)),
                index=returns.index,
                name=self.benchmark_symbol
            )
        else:
            benchmark = benchmark.loc[common_dates]
            returns = returns.loc[common_dates]
        
        print(f"📊 Generating QuantStats tearsheet for {strategy_name}...")
        print(f"   Period: {returns.index.min().date()} to {returns.index.max().date()}")
        print(f"   Trading days: {len(returns)}")
        
        # Generate tearsheet
        if output_file:
            qs.reports.html(returns, benchmark=benchmark, output=output_file, title=strategy_name)
            print(f"📄 Tearsheet saved to: {output_file}")
        else:
            qs.reports.full(returns, benchmark=benchmark, title=strategy_name)
    
    def get_key_metrics(self, backtest_results: Dict[str, Any],
                       benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Get key performance metrics using QuantStats
        
        Args:
            backtest_results: Results from backtester
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of key metrics
        """
        returns = self.prepare_returns_series(backtest_results)
        benchmark = self.load_benchmark_data(benchmark_returns)
        
        # Align data
        common_dates = returns.index.intersection(benchmark.index)
        if len(common_dates) > 0:
            benchmark = benchmark.loc[common_dates]
            returns = returns.loc[common_dates]
        
        metrics = {
            # Return metrics
            'Total Return': qs.stats.comp(returns),
            'Annual Return': qs.stats.cagr(returns),
            'Volatility': qs.stats.volatility(returns),
            
            # Risk-adjusted metrics
            'Sharpe Ratio': qs.stats.sharpe(returns),
            'Sortino Ratio': qs.stats.sortino(returns),
            'Calmar Ratio': qs.stats.calmar(returns),
            
            # Drawdown metrics
            'Max Drawdown': qs.stats.max_drawdown(returns),
            'Avg Drawdown': qs.stats.avg_drawdown(returns),
            'Recovery Factor': qs.stats.recovery_factor(returns),
            
            # Risk metrics
            'VaR (95%)': qs.stats.var(returns),
            'CVaR (95%)': qs.stats.cvar(returns),
            'Skewness': qs.stats.skew(returns),
            'Kurtosis': qs.stats.kurtosis(returns),
            
            # Benchmark comparison (if available)
            'Beta': qs.stats.beta(returns, benchmark) if len(benchmark) > 0 else np.nan,
            'Alpha': qs.stats.alpha(returns, benchmark) if len(benchmark) > 0 else np.nan,
            'Information Ratio': qs.stats.information_ratio(returns, benchmark) if len(benchmark) > 0 else np.nan,
        }
        
        return metrics
    
    def plot_performance_charts(self, backtest_results: Dict[str, Any],
                               strategy_name: str = "Strategy",
                               save_path: Optional[str] = None) -> None:
        """
        Generate performance charts using QuantStats
        
        Args:
            backtest_results: Results from backtester
            strategy_name: Name of the strategy
            save_path: Path to save charts (optional)
        """
        returns = self.prepare_returns_series(backtest_results)
        benchmark = self.load_benchmark_data()
        
        # Align data
        common_dates = returns.index.intersection(benchmark.index)
        if len(common_dates) > 0:
            benchmark = benchmark.loc[common_dates]
            returns = returns.loc[common_dates]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} Performance Analysis', fontsize=16)
        
        # Cumulative returns
        qs.plots.returns(returns, benchmark=benchmark, ax=axes[0, 0])
        axes[0, 0].set_title('Cumulative Returns')
        
        # Rolling Sharpe
        qs.plots.rolling_sharpe(returns, ax=axes[0, 1])
        axes[0, 1].set_title('Rolling Sharpe Ratio')
        
        # Drawdown
        qs.plots.drawdown(returns, ax=axes[1, 0])
        axes[1, 0].set_title('Drawdown')
        
        # Monthly returns heatmap
        qs.plots.monthly_heatmap(returns, ax=axes[1, 1])
        axes[1, 1].set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Charts saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_report(self, strategy_results: Dict[str, Dict[str, Any]],
                               output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Create comparison report for multiple strategies
        
        Args:
            strategy_results: Dictionary mapping strategy names to backtest results
            output_file: Path to save comparison report
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = {}
        
        for strategy_name, results in strategy_results.items():
            try:
                metrics = self.get_key_metrics(results)
                comparison_data[strategy_name] = metrics
            except Exception as e:
                print(f"⚠️  Error analyzing {strategy_name}: {e}")
                continue
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Sort by Sharpe ratio
        if 'Sharpe Ratio' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
        
        if output_file:
            comparison_df.to_csv(output_file)
            print(f"📊 Comparison report saved to: {output_file}")
        
        return comparison_df


# Convenience functions for easy usage
def generate_tearsheet(backtest_results: Dict[str, Any], 
                      strategy_name: str = "Strategy",
                      output_file: Optional[str] = None,
                      benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Quick tearsheet generation
    
    Args:
        backtest_results: Results from backtester
        strategy_name: Name of the strategy
        output_file: Path to save HTML report
        benchmark_returns: Benchmark returns for comparison
        
    Returns:
        Dictionary of key metrics
    """
    analyzer = QuantStatsAnalyzer()
    
    # Generate tearsheet
    analyzer.generate_full_tearsheet(
        backtest_results, 
        strategy_name, 
        output_file, 
        benchmark_returns
    )
    
    # Return key metrics
    return analyzer.get_key_metrics(backtest_results, benchmark_returns)


def compare_strategies_tearsheet(strategy_results: Dict[str, Dict[str, Any]],
                                output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Generate tearsheets for multiple strategies and create comparison
    
    Args:
        strategy_results: Dictionary mapping strategy names to backtest results
        output_dir: Directory to save individual tearsheets
        
    Returns:
        DataFrame with comparison metrics
    """
    analyzer = QuantStatsAnalyzer()
    
    # Generate individual tearsheets
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for strategy_name, results in strategy_results.items():
            output_file = Path(output_dir) / f"{strategy_name}_tearsheet.html"
            try:
                analyzer.generate_full_tearsheet(results, strategy_name, str(output_file))
            except Exception as e:
                print(f"⚠️  Error generating tearsheet for {strategy_name}: {e}")
    
    # Create comparison report
    comparison_file = Path(output_dir) / "strategy_comparison.csv" if output_dir else None
    return analyzer.create_comparison_report(strategy_results, comparison_file) 