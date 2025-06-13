"""Backtesting Analytics Module - Phase 3 Advanced Analytics"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class BacktestAnalyzer:
    """Comprehensive backtesting analysis across strategies and time periods"""
    
    def __init__(self):
        self.backtest_results = {}
        self.analysis_cache = {}
    
    def add_backtest_result(self, 
                           strategy_name: str,
                           backtest_data: Dict[str, Any],
                           metadata: Dict[str, Any] = None):
        """Add backtest result for analysis"""
        self.backtest_results[strategy_name] = {
            'data': backtest_data,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
    
    def analyze_all_strategies(self) -> Dict[str, Any]:
        """Comprehensive analysis across all strategies"""
        if not self.backtest_results:
            return {}
        
        # Strategy comparison
        strategy_comparison = self._compare_strategies()
        
        # Performance ranking
        performance_ranking = self._rank_strategies()
        
        # Risk-return analysis
        risk_return_analysis = self._analyze_risk_return()
        
        # Consistency analysis
        consistency_analysis = self._analyze_consistency()
        
        return {
            'strategy_comparison': strategy_comparison,
            'performance_ranking': performance_ranking,
            'risk_return_analysis': risk_return_analysis,
            'consistency_analysis': consistency_analysis,
            'analysis_summary': self._create_analysis_summary()
        }
    
    def _compare_strategies(self) -> pd.DataFrame:
        """Compare strategies across key metrics"""
        comparison_data = {}
        
        for strategy_name, result in self.backtest_results.items():
            data = result['data']
            
            if isinstance(data, dict):
                metrics = {
                    'Total Return': data.get('total_return', 0),
                    'Sharpe Ratio': data.get('sharpe_ratio', 0),
                    'Max Drawdown': data.get('max_drawdown', 0),
                    'Volatility': data.get('volatility', 0),
                    'Win Rate': data.get('win_rate', 0)
                }
                
                comparison_data[strategy_name] = metrics
        
        return pd.DataFrame(comparison_data).T
    
    def _rank_strategies(self) -> Dict[str, pd.DataFrame]:
        """Rank strategies by different metrics"""
        comparison_df = self._compare_strategies()
        
        if comparison_df.empty:
            return {}
        
        rankings = {}
        ranking_metrics = ['Total Return', 'Sharpe Ratio']
        
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                ranked = comparison_df.sort_values(metric, ascending=False)
                ranked['Rank'] = range(1, len(ranked) + 1)
                rankings[metric] = ranked[['Rank', metric]]
        
        return rankings
    
    def _analyze_risk_return(self) -> Dict[str, Any]:
        """Analyze risk-return characteristics"""
        comparison_df = self._compare_strategies()
        
        if comparison_df.empty or 'Total Return' not in comparison_df.columns:
            return {}
        
        returns = comparison_df['Total Return']
        
        return {
            'return_statistics': {
                'mean': returns.mean(),
                'std': returns.std(),
                'min': returns.min(),
                'max': returns.max()
            }
        }
    
    def _analyze_consistency(self) -> Dict[str, Any]:
        """Analyze strategy consistency"""
        consistency_metrics = {}
        
        for strategy_name, result in self.backtest_results.items():
            data = result['data']
            
            if isinstance(data, dict):
                # Basic consistency metrics
                total_return = data.get('total_return', 0)
                volatility = data.get('volatility', 0)
                max_drawdown = abs(data.get('max_drawdown', 0))
                
                # Consistency score (penalize high volatility and drawdowns)
                consistency_score = total_return / (1 + volatility + max_drawdown) if (volatility + max_drawdown) > 0 else 0
                
                consistency_metrics[strategy_name] = {
                    'consistency_score': consistency_score,
                    'return_volatility_ratio': total_return / volatility if volatility > 0 else 0,
                    'drawdown_penalty': max_drawdown
                }
        
        return consistency_metrics
    
    def _create_analysis_summary(self) -> Dict[str, Any]:
        """Create high-level analysis summary"""
        comparison_df = self._compare_strategies()
        
        if comparison_df.empty:
            return {}
        
        best_return = comparison_df['Total Return'].idxmax() if 'Total Return' in comparison_df.columns else None
        best_sharpe = comparison_df['Sharpe Ratio'].idxmax() if 'Sharpe Ratio' in comparison_df.columns else None
        
        return {
            'total_strategies_analyzed': len(self.backtest_results),
            'best_return_strategy': best_return,
            'best_sharpe_strategy': best_sharpe,
            'analysis_date': datetime.now()
        }


class MultiStrategyAnalyzer:
    """Analyze multiple strategies running simultaneously"""
    
    def __init__(self):
        self.strategy_results = {}
    
    def add_strategy_result(self, name: str, returns: pd.Series):
        """Add strategy result for multi-strategy analysis"""
        self.strategy_results[name] = {'returns': returns}
    
    def analyze_portfolio_combination(self, weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze combination of strategies as portfolio"""
        if not self.strategy_results:
            return {}
        
        # Default equal weights
        if weights is None:
            n_strategies = len(self.strategy_results)
            weights = {name: 1.0/n_strategies for name in self.strategy_results.keys()}
        
        # Calculate combined portfolio returns
        portfolio_returns = self._calculate_combined_returns(weights)
        
        # Portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_returns)
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_metrics': portfolio_metrics,
            'weights': weights
        }
    
    def _calculate_combined_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Calculate combined portfolio returns"""
        return_series = []
        strategy_names = []
        
        for name, data in self.strategy_results.items():
            if name in weights:
                return_series.append(data['returns'])
                strategy_names.append(name)
        
        if not return_series:
            return pd.Series()
        
        # Combine returns
        aligned_returns = pd.concat(return_series, axis=1, join='inner')
        aligned_returns.columns = strategy_names
        
        # Calculate weighted returns
        weight_vector = [weights[name] for name in strategy_names]
        portfolio_returns = (aligned_returns * weight_vector).sum(axis=1)
        
        return portfolio_returns
    
    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if len(returns) == 0:
            return {}
        
        return {
            'total_return': (1 + returns).prod() - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class PeriodAnalyzer:
    """Analyze strategy performance across different time periods"""
    
    def analyze_periods(self, 
                       returns: pd.Series,
                       period_type: str = 'monthly') -> Dict[str, Any]:
        """Analyze performance by time periods"""
        
        if period_type == 'monthly':
            period_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        elif period_type == 'yearly':
            period_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        else:
            return {}
        
        return {
            'period_returns': period_returns,
            'period_statistics': {
                'mean': period_returns.mean(),
                'std': period_returns.std(),
                'best_period': period_returns.max(),
                'worst_period': period_returns.min(),
                'positive_periods': (period_returns > 0).sum(),
                'win_rate': (period_returns > 0).mean()
            }
        }


class SectorAnalyzer:
    """Analyze strategy performance across different market sectors"""
    
    def __init__(self):
        self.sector_data = {}
    
    def add_sector_data(self, sector: str, returns: pd.Series):
        """Add sector performance data"""
        self.sector_data[sector] = returns
    
    def analyze_sector_performance(self, strategy_returns: pd.Series) -> Dict[str, Any]:
        """Analyze strategy performance across sectors"""
        sector_analysis = {}
        
        for sector, sector_returns in self.sector_data.items():
            # Align returns
            aligned_data = pd.concat([strategy_returns, sector_returns], axis=1, join='inner')
            aligned_data.columns = ['strategy', 'sector']
            
            if len(aligned_data) > 10:
                # Calculate correlation
                correlation = aligned_data['strategy'].corr(aligned_data['sector'])
                
                sector_analysis[sector] = {
                    'correlation': correlation,
                    'sector_return': (1 + aligned_data['sector']).prod() - 1,
                    'strategy_return_in_sector': (1 + aligned_data['strategy']).prod() - 1
                }
        
        return sector_analysis 