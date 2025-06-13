"""Portfolio Analytics Module - Phase 3 Advanced Analytics"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class PortfolioAnalyzer:
    """Comprehensive portfolio analysis and optimization"""
    
    def __init__(self):
        self.portfolio_data = {}
        self.correlation_analyzer = CorrelationAnalyzer()
    
    def add_strategy(self, name: str, returns: pd.Series, weight: float = 1.0):
        """Add strategy to portfolio analysis"""
        self.portfolio_data[name] = {
            'returns': returns,
            'weight': weight
        }
    
    def analyze_portfolio(self) -> Dict[str, Any]:
        """Comprehensive portfolio analysis"""
        if not self.portfolio_data:
            return {}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        
        # Individual strategy analysis
        strategy_analysis = self._analyze_individual_strategies()
        
        # Correlation analysis
        correlation_analysis = self.correlation_analyzer.analyze_correlations(
            {name: data['returns'] for name, data in self.portfolio_data.items()}
        )
        
        # Portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_returns)
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_metrics': portfolio_metrics,
            'strategy_analysis': strategy_analysis,
            'correlation_analysis': correlation_analysis
        }
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate weighted portfolio returns"""
        if not self.portfolio_data:
            return pd.Series()
        
        # Get return series and weights
        return_series = [data['returns'] for data in self.portfolio_data.values()]
        weights = [data['weight'] for data in self.portfolio_data.values()]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Align series and calculate weighted returns
        aligned_returns = pd.concat(return_series, axis=1, join='inner')
        aligned_returns.columns = list(self.portfolio_data.keys())
        
        # Calculate weighted portfolio returns
        portfolio_returns = (aligned_returns * normalized_weights).sum(axis=1)
        
        return portfolio_returns
    
    def _analyze_individual_strategies(self) -> Dict[str, Dict]:
        """Analyze individual strategy performance"""
        strategy_analysis = {}
        
        for name, data in self.portfolio_data.items():
            returns = data['returns']
            
            strategy_analysis[name] = {
                'total_return': (1 + returns).prod() - 1,
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'weight': data['weight']
            }
        
        return strategy_analysis
    
    def _calculate_portfolio_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        if len(portfolio_returns) == 0:
            return {}
        
        return {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class CorrelationAnalyzer:
    """Strategy correlation analysis"""
    
    def analyze_correlations(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze correlations between strategies"""
        if len(strategy_returns) < 2:
            return {}
        
        # Align all return series
        returns_df = pd.concat(strategy_returns.values(), axis=1, join='inner', keys=strategy_returns.keys())
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Correlation statistics
        correlation_stats = self._analyze_correlation_statistics(correlation_matrix)
        
        return {
            'correlation_matrix': correlation_matrix,
            'correlation_statistics': correlation_stats,
            'diversification_potential': self._assess_diversification_potential(correlation_matrix)
        }
    
    def _analyze_correlation_statistics(self, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlation matrix statistics"""
        # Extract upper triangle (excluding diagonal)
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_triangle.stack().values
        
        return {
            'average_correlation': np.mean(correlations),
            'max_correlation': np.max(correlations),
            'min_correlation': np.min(correlations)
        }
    
    def _assess_diversification_potential(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Assess diversification potential based on correlations"""
        # Extract correlations
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_triangle.stack().values
        
        # Count correlations in different ranges
        low_corr = np.sum(correlations < 0.3)
        high_corr = np.sum(correlations >= 0.7)
        total_pairs = len(correlations)
        
        return {
            'low_correlation_pairs': low_corr,
            'high_correlation_pairs': high_corr,
            'diversification_score': low_corr / total_pairs if total_pairs > 0 else 0
        }


class RegimeDetector:
    """Market regime detection and analysis"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_model = None
        self.scaler = StandardScaler()
    
    def detect_regimes(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regimes using clustering"""
        # Prepare features
        feature_data = self._prepare_features(market_data)
        
        if feature_data.empty:
            return {}
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Fit clustering model
        self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42)
        regime_labels = self.regime_model.fit_predict(scaled_features)
        
        # Create regime series
        regime_series = pd.Series(regime_labels, index=feature_data.index)
        
        # Analyze regimes
        regime_analysis = self._analyze_regimes(market_data, regime_series)
        
        return {
            'regime_labels': regime_series,
            'regime_analysis': regime_analysis
        }
    
    def _prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection"""
        feature_data = pd.DataFrame(index=market_data.index)
        
        if 'Close' in market_data.columns:
            feature_data['returns'] = market_data['Close'].pct_change()
            feature_data['volatility'] = feature_data['returns'].rolling(21).std()
        
        # Drop NaN values
        feature_data = feature_data.dropna()
        
        return feature_data
    
    def _analyze_regimes(self, market_data: pd.DataFrame, regime_series: pd.Series) -> Dict[int, Dict]:
        """Analyze characteristics of each regime"""
        regime_analysis = {}
        
        if 'Close' in market_data.columns:
            returns = market_data['Close'].pct_change()
            
            # Align returns and regime_series indices
            aligned_returns, aligned_regimes = returns.align(regime_series, join='inner')
            
            for regime in range(self.n_regimes):
                regime_mask = aligned_regimes == regime
                regime_returns = aligned_returns[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_analysis[regime] = {
                        'count': len(regime_returns),
                        'percentage': len(regime_returns) / len(aligned_regimes) * 100,
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std()
                    }
        
        return regime_analysis


class FactorAnalyzer:
    """Factor analysis for strategy performance"""
    
    def analyze_factors(self, 
                       strategy_returns: Dict[str, pd.Series],
                       factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze factor exposures of strategies"""
        factor_analysis = {}
        
        for strategy_name, returns in strategy_returns.items():
            # Align returns with factors
            aligned_data = pd.concat([returns] + list(factor_returns.values()), 
                                   axis=1, join='inner')
            aligned_data.columns = ['strategy'] + list(factor_returns.keys())
            
            # Run factor regression
            factor_result = self._run_factor_regression(aligned_data)
            factor_analysis[strategy_name] = factor_result
        
        return factor_analysis
    
    def _run_factor_regression(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run factor regression analysis"""
        if len(data) < 10:
            return {}
        
        y = data['strategy']
        X = data.drop('strategy', axis=1)
        
        # Add constant term
        X = pd.concat([pd.Series(1, index=X.index, name='alpha'), X], axis=1)
        
        # Calculate factor loadings using linear regression
        try:
            # Convert to numpy arrays for matrix operations
            X_np = X.values
            y_np = y.values
            
            # Calculate beta coefficients
            XtX_inv = np.linalg.inv(X_np.T @ X_np)
            beta = XtX_inv @ X_np.T @ y_np
            
            # Calculate R-squared
            y_pred = X_np @ beta
            residuals = y_np - y_pred
            tss = ((y_np - y_np.mean()) ** 2).sum()
            rss = (residuals ** 2).sum()
            r_squared = 1 - (rss / tss) if tss != 0 else 0
            
            return {
                'alpha': beta[0],
                'factor_loadings': dict(zip(X.columns[1:], beta[1:])),
                'r_squared': r_squared
            }
            
        except (np.linalg.LinAlgError, ValueError) as e:
            return {'error': f'Cannot compute factor loadings: {str(e)}'}


class PortfolioOptimizer:
    """Portfolio optimization using various methods"""
    
    def optimize_portfolio(self, 
                          strategy_returns: Dict[str, pd.Series],
                          method: str = 'equal_weight') -> Dict[str, Any]:
        """Optimize portfolio weights"""
        # Align return series
        returns_df = pd.concat(strategy_returns.values(), axis=1, join='inner', 
                              keys=strategy_returns.keys())
        
        if method == 'equal_weight':
            return self._equal_weight_optimization(returns_df)
        elif method == 'risk_parity':
            return self._risk_parity_optimization(returns_df)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _equal_weight_optimization(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Equal weight portfolio optimization"""
        n_assets = len(returns_df.columns)
        weights = np.ones(n_assets) / n_assets
        
        return {
            'weights': dict(zip(returns_df.columns, weights)),
            'method': 'equal_weight'
        }
    
    def _risk_parity_optimization(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Risk parity portfolio optimization"""
        # Calculate volatilities
        volatilities = returns_df.std()
        
        # Inverse volatility weights
        inv_vol_weights = 1 / volatilities
        weights = inv_vol_weights / inv_vol_weights.sum()
        
        return {
            'weights': dict(zip(returns_df.columns, weights)),
            'method': 'risk_parity'
        } 