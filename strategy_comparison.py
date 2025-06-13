"""
Strategy Comparison Framework
Compare multiple trading strategies side by side with detailed analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Import backtesting components
from ..core.backtester import Backtester
from ..data.data_loader import DataLoader


class StrategyComparison:
    """
    Compare multiple trading strategies with comprehensive analysis
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize strategy comparison
        
        Args:
            initial_capital: Starting capital for each strategy
        """
        self.initial_capital = initial_capital
        self.strategies = {}
        self.results = {}
        self.comparison_metrics = {}
        
    def add_strategy(self, name: str, strategy_func: Callable, description: str = ""):
        """
        Add a strategy to compare
        
        Args:
            name: Strategy name
            strategy_func: Strategy function
            description: Strategy description
        """
        self.strategies[name] = {
            'function': strategy_func,
            'description': description
        }
        
    def run_comparison(self, data_loader: DataLoader, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      benchmark_symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest comparison for all strategies
        
        Args:
            data_loader: DataLoader with market data
            start_date: Start date for backtest
            end_date: End date for backtest
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            Comparison results dictionary
        """
        print("🚀 RUNNING STRATEGY COMPARISON")
        print("=" * 60)
        
        for strategy_name, strategy_info in self.strategies.items():
            print(f"\n📈 Testing {strategy_name}...")
            
            # Create fresh backtester for each strategy
            backtester = Backtester(initial_capital=self.initial_capital)
            backtester.load_data(data_loader)
            backtester.set_strategy(strategy_info['function'])
            
            # Run backtest
            try:
                results = backtester.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    benchmark_symbol=benchmark_symbol
                )
                
                self.results[strategy_name] = results
                print(f"✅ {strategy_name} completed: {results['metrics']['total_return']:.2%} return")
                
            except Exception as e:
                print(f"❌ {strategy_name} failed: {str(e)}")
                self.results[strategy_name] = {'error': str(e)}
        
        # Generate comparison metrics
        self._generate_comparison_metrics()
        
        return self.results
    
    def _generate_comparison_metrics(self):
        """Generate comparative performance metrics"""
        valid_results = {name: result for name, result in self.results.items() 
                        if 'error' not in result}
        
        if not valid_results:
            return
        
        # Extract key metrics for comparison
        comparison_data = {}
        
        for strategy_name, result in valid_results.items():
            metrics = result['metrics']
            comparison_data[strategy_name] = {
                'Total Return': metrics['total_return'],
                'Final Value': metrics['final_value'],
                'Total Trades': metrics['total_trades'],
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Best Day': metrics.get('best_day', 0),
                'Worst Day': metrics.get('worst_day', 0),
                'Win Rate': self._calculate_win_rate(result),
                'Avg Trade Return': self._calculate_avg_trade_return(result),
                'Profit Factor': self._calculate_profit_factor(result)
            }
        
        self.comparison_metrics = pd.DataFrame(comparison_data).T
        
        # Add rankings
        self._add_rankings()
    
    def _calculate_win_rate(self, result: Dict[str, Any]) -> float:
        """Calculate win rate from trade history"""
        if 'trade_history' not in result or result['trade_history'].empty:
            return 0.0
        
        trades = result['trade_history']
        if 'pnl' in trades.columns:
            winning_trades = len(trades[trades['pnl'] > 0])
            total_trades = len(trades)
            return winning_trades / max(1, total_trades)
        
        return 0.0
    
    def _calculate_avg_trade_return(self, result: Dict[str, Any]) -> float:
        """Calculate average trade return"""
        if 'trade_history' not in result or result['trade_history'].empty:
            return 0.0
        
        trades = result['trade_history']
        if 'pnl' in trades.columns:
            return trades['pnl'].mean()
        
        return 0.0
    
    def _calculate_profit_factor(self, result: Dict[str, Any]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if 'trade_history' not in result or result['trade_history'].empty:
            return 0.0
        
        trades = result['trade_history']
        if 'pnl' in trades.columns:
            gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            
            if gross_loss > 0:
                return gross_profit / gross_loss
        
        return 0.0
    
    def _add_rankings(self):
        """Add performance rankings"""
        if self.comparison_metrics.empty:
            return
        
        # Rank strategies by key metrics (higher is better except for drawdown)
        rankings = pd.DataFrame(index=self.comparison_metrics.index)
        
        # Higher is better
        for metric in ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Profit Factor']:
            if metric in self.comparison_metrics.columns:
                rankings[f'{metric} Rank'] = self.comparison_metrics[metric].rank(ascending=False)
        
        # Lower is better
        for metric in ['Max Drawdown', 'Volatility']:
            if metric in self.comparison_metrics.columns:
                rankings[f'{metric} Rank'] = self.comparison_metrics[metric].rank(ascending=True)
        
        # Calculate overall score (average of ranks)
        rank_columns = [col for col in rankings.columns if 'Rank' in col]
        if rank_columns:
            rankings['Overall Score'] = rankings[rank_columns].mean(axis=1)
            rankings['Overall Rank'] = rankings['Overall Score'].rank(ascending=True)
        
        self.rankings = rankings
    
    def print_comparison(self):
        """Print formatted comparison results"""
        print("\n" + "=" * 80)
        print("📊 STRATEGY COMPARISON RESULTS")
        print("=" * 80)
        
        if self.comparison_metrics.empty:
            print("❌ No valid results to compare")
            return
        
        # Performance Summary
        print("\n📈 PERFORMANCE SUMMARY:")
        print("-" * 60)
        
        display_metrics = self.comparison_metrics.copy()
        
        # Format percentages
        for col in ['Total Return', 'Volatility', 'Max Drawdown', 'Best Day', 'Worst Day', 'Win Rate']:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2%}")
        
        # Format currency
        display_metrics['Final Value'] = display_metrics['Final Value'].apply(lambda x: f"${x:,.0f}")
        
        # Format ratios
        for col in ['Sharpe Ratio', 'Profit Factor']:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2f}")
        
        print(display_metrics.to_string())
        
        # Rankings
        if hasattr(self, 'rankings') and not self.rankings.empty:
            print("\n🏆 STRATEGY RANKINGS:")
            print("-" * 40)
            
            if 'Overall Rank' in self.rankings.columns:
                sorted_strategies = self.rankings.sort_values('Overall Rank')
                for i, (strategy, row) in enumerate(sorted_strategies.iterrows(), 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    print(f"{medal} {strategy} (Score: {row['Overall Score']:.1f})")
        
        # Best Strategy Highlight
        if 'Total Return' in self.comparison_metrics.columns:
            best_return_strategy = self.comparison_metrics['Total Return'].idxmax()
            best_return = self.comparison_metrics.loc[best_return_strategy, 'Total Return']
            
            print(f"\n🎯 BEST PERFORMER: {best_return_strategy}")
            print(f"   Total Return: {best_return:.2%}")
            
            if best_return_strategy in self.strategies:
                description = self.strategies[best_return_strategy].get('description', '')
                if description:
                    print(f"   Description: {description}")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Plot strategy comparison charts
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.comparison_metrics.empty:
            print("❌ No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Total Returns
        ax1 = axes[0, 0]
        returns = self.comparison_metrics['Total Return'] * 100
        bars1 = ax1.bar(range(len(returns)), returns.values)
        ax1.set_title('Total Returns (%)')
        ax1.set_xticks(range(len(returns)))
        ax1.set_xticklabels(returns.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Color bars based on performance
        for i, bar in enumerate(bars1):
            if returns.iloc[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # 2. Risk-Adjusted Returns (Sharpe Ratio)
        ax2 = axes[0, 1]
        if 'Sharpe Ratio' in self.comparison_metrics.columns:
            sharpe = self.comparison_metrics['Sharpe Ratio']
            bars2 = ax2.bar(range(len(sharpe)), sharpe.values)
            ax2.set_title('Sharpe Ratio')
            ax2.set_xticks(range(len(sharpe)))
            ax2.set_xticklabels(sharpe.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Color bars
            for i, bar in enumerate(bars2):
                if sharpe.iloc[i] > 1:
                    bar.set_color('darkgreen')
                elif sharpe.iloc[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
        
        # 3. Maximum Drawdown
        ax3 = axes[1, 0]
        if 'Max Drawdown' in self.comparison_metrics.columns:
            drawdown = self.comparison_metrics['Max Drawdown'] * 100
            bars3 = ax3.bar(range(len(drawdown)), drawdown.values)
            ax3.set_title('Maximum Drawdown (%)')
            ax3.set_xticks(range(len(drawdown)))
            ax3.set_xticklabels(drawdown.index, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # All drawdowns should be red (negative)
            for bar in bars3:
                bar.set_color('red')
        
        # 4. Win Rate
        ax4 = axes[1, 1]
        if 'Win Rate' in self.comparison_metrics.columns:
            win_rate = self.comparison_metrics['Win Rate'] * 100
            bars4 = ax4.bar(range(len(win_rate)), win_rate.values)
            ax4.set_title('Win Rate (%)')
            ax4.set_xticks(range(len(win_rate)))
            ax4.set_xticklabels(win_rate.index, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)
            
            # Color based on win rate
            for i, bar in enumerate(bars4):
                if win_rate.iloc[i] > 60:
                    bar.set_color('darkgreen')
                elif win_rate.iloc[i] > 50:
                    bar.set_color('green')
                else:
                    bar.set_color('orange')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to {save_path}")
        
        plt.show()
    
    def get_best_strategy(self) -> Optional[str]:
        """Get the name of the best performing strategy"""
        if hasattr(self, 'rankings') and not self.rankings.empty and 'Overall Rank' in self.rankings.columns:
            return self.rankings['Overall Rank'].idxmin()
        elif not self.comparison_metrics.empty and 'Total Return' in self.comparison_metrics.columns:
            return self.comparison_metrics['Total Return'].idxmax()
        return None
    
    def export_results(self, filepath: str):
        """Export comparison results to Excel"""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main comparison metrics
                if not self.comparison_metrics.empty:
                    self.comparison_metrics.to_excel(writer, sheet_name='Performance Metrics')
                
                # Rankings
                if hasattr(self, 'rankings') and not self.rankings.empty:
                    self.rankings.to_excel(writer, sheet_name='Rankings')
                
                # Individual strategy details
                for strategy_name, result in self.results.items():
                    if 'error' not in result and 'trade_history' in result:
                        trade_history = result['trade_history']
                        if not trade_history.empty:
                            sheet_name = f"{strategy_name[:25]}_Trades"  # Excel sheet name limit
                            trade_history.to_excel(writer, sheet_name=sheet_name)
            
            print(f"📊 Results exported to {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to export results: {str(e)}")


# Convenience function for quick comparisons
def compare_strategies(strategies: Dict[str, Callable], 
                      data_loader: DataLoader,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      initial_capital: float = 100000) -> StrategyComparison:
    """
    Quick strategy comparison function
    
    Args:
        strategies: Dictionary of {name: strategy_function}
        data_loader: DataLoader with market data
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        
    Returns:
        StrategyComparison object with results
    """
    comparison = StrategyComparison(initial_capital)
    
    for name, strategy_func in strategies.items():
        comparison.add_strategy(name, strategy_func)
    
    comparison.run_comparison(data_loader, start_date, end_date)
    comparison.print_comparison()
    
    return comparison 