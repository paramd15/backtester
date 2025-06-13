"""Monte Carlo Simulation Module - Phase 3 Advanced Analytics"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
import random


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy performance analysis"""
    
    def __init__(self, n_simulations: int = 1000, random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        if random_seed:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.simulation_results = []
    
    def run_simulations(self, 
                       strategy_function: Callable,
                       data: pd.DataFrame,
                       parameter_distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Run Monte Carlo simulations with parameter uncertainty"""
        print(f"Running {self.n_simulations} Monte Carlo simulations...")
        
        for i in range(self.n_simulations):
            # Sample parameters
            sampled_params = self._sample_parameters(parameter_distributions)
            
            # Run strategy simulation
            try:
                result = strategy_function(data, sampled_params)
                
                self.simulation_results.append({
                    'simulation': i,
                    'parameters': sampled_params,
                    'results': result
                })
                
            except Exception as e:
                print(f"Simulation {i} failed: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.n_simulations} simulations")
        
        return self._analyze_simulation_results()
    
    def _sample_parameters(self, parameter_distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Sample parameters from specified distributions"""
        sampled_params = {}
        
        for param_name, distribution in parameter_distributions.items():
            dist_type = distribution.get('type', 'uniform')
            
            if dist_type == 'uniform':
                min_val = distribution['min']
                max_val = distribution['max']
                sampled_params[param_name] = np.random.uniform(min_val, max_val)
                
            elif dist_type == 'normal':
                mean = distribution['mean']
                std = distribution['std']
                sampled_params[param_name] = np.random.normal(mean, std)
                
            elif dist_type == 'choice':
                choices = distribution['choices']
                sampled_params[param_name] = np.random.choice(choices)
                
            else:
                sampled_params[param_name] = distribution.get('mean', 0)
        
        return sampled_params
    
    def _analyze_simulation_results(self) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        if not self.simulation_results:
            return {}
        
        # Extract performance metrics
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for sim_result in self.simulation_results:
            results = sim_result['results']
            
            if isinstance(results, dict):
                returns.append(results.get('total_return', 0))
                sharpe_ratios.append(results.get('sharpe_ratio', 0))
                max_drawdowns.append(results.get('max_drawdown', 0))
        
        # Calculate statistics
        return_stats = self._calculate_statistics(returns)
        sharpe_stats = self._calculate_statistics(sharpe_ratios)
        drawdown_stats = self._calculate_statistics(max_drawdowns)
        
        return {
            'n_simulations': len(self.simulation_results),
            'return_statistics': return_stats,
            'sharpe_statistics': sharpe_stats,
            'drawdown_statistics': drawdown_stats,
            'confidence_intervals': self._calculate_confidence_intervals(returns),
            'risk_metrics': self._calculate_risk_metrics(returns)
        }
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {}
        
        values_array = np.array(values)
        
        return {
            'mean': np.mean(values_array),
            'median': np.median(values_array),
            'std': np.std(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array),
            'percentile_5': np.percentile(values_array, 5),
            'percentile_95': np.percentile(values_array, 95)
        }
    
    def _calculate_confidence_intervals(self, returns: List[float]) -> Dict[str, float]:
        """Calculate confidence intervals for returns"""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        return {
            'ci_95_lower': np.percentile(returns_array, 2.5),
            'ci_95_upper': np.percentile(returns_array, 97.5),
            'ci_99_lower': np.percentile(returns_array, 0.5),
            'ci_99_upper': np.percentile(returns_array, 99.5)
        }
    
    def _calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate risk metrics from simulation results"""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Probability of loss
        prob_loss = np.sum(returns_array < 0) / len(returns_array)
        
        # Expected shortfall
        losses = returns_array[returns_array < 0]
        expected_shortfall = np.mean(losses) if len(losses) > 0 else 0
        
        return {
            'probability_of_loss': prob_loss,
            'expected_shortfall': expected_shortfall,
            'worst_case_scenario': np.min(returns_array),
            'best_case_scenario': np.max(returns_array)
        }


class StressTestSimulator:
    """Stress testing for extreme market scenarios"""
    
    def __init__(self):
        self.stress_scenarios = []
    
    def add_stress_scenario(self, 
                           name: str,
                           market_shock: Dict[str, float],
                           description: str = ""):
        """Add a stress test scenario"""
        self.stress_scenarios.append({
            'name': name,
            'market_shock': market_shock,
            'description': description
        })
    
    def run_stress_tests(self, 
                        strategy_function: Callable,
                        base_data: pd.DataFrame,
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress tests on strategy"""
        print(f"Running {len(self.stress_scenarios)} stress test scenarios...")
        
        stress_results = {}
        
        for scenario in self.stress_scenarios:
            print(f"Testing scenario: {scenario['name']}")
            
            # Apply market shock to data
            shocked_data = self._apply_market_shock(base_data, scenario['market_shock'])
            
            # Run strategy on shocked data
            try:
                result = strategy_function(shocked_data, parameters)
                stress_results[scenario['name']] = {
                    'scenario': scenario,
                    'results': result
                }
                
            except Exception as e:
                print(f"Stress test {scenario['name']} failed: {e}")
                stress_results[scenario['name']] = {
                    'scenario': scenario,
                    'results': {'error': str(e)}
                }
        
        return stress_results
    
    def _apply_market_shock(self, data: pd.DataFrame, market_shock: Dict[str, float]) -> pd.DataFrame:
        """Apply market shock to data"""
        shocked_data = data.copy()
        
        # Apply return shock
        if 'return_shock' in market_shock:
            shock_magnitude = market_shock['return_shock']
            
            if 'Close' in shocked_data.columns:
                shocked_data.loc[shocked_data.index[0], 'Close'] *= (1 + shock_magnitude)
                
                # Recalculate subsequent prices
                for i in range(1, len(shocked_data)):
                    prev_close = shocked_data.iloc[i-1]['Close']
                    original_return = (data.iloc[i]['Close'] / data.iloc[i-1]['Close']) - 1
                    shocked_data.iloc[i, shocked_data.columns.get_loc('Close')] = prev_close * (1 + original_return)
        
        return shocked_data
    
    def create_standard_scenarios(self):
        """Create standard stress test scenarios"""
        self.add_stress_scenario(
            "Market Crash",
            {'return_shock': -0.20},
            "20% market decline"
        )
        
        self.add_stress_scenario(
            "Bear Market",
            {'return_shock': -0.10},
            "10% market decline"
        )
        
        self.add_stress_scenario(
            "Flash Crash",
            {'return_shock': -0.15},
            "15% sudden decline"
        )


class BootstrapSimulator:
    """Bootstrap simulation for resampling historical data"""
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.bootstrap_results = []
    
    def run_bootstrap(self, 
                     returns: pd.Series,
                     strategy_function: Callable,
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run bootstrap simulation by resampling historical returns"""
        print(f"Running {self.n_simulations} bootstrap simulations...")
        
        for i in range(self.n_simulations):
            # Create bootstrap sample
            bootstrap_returns = returns.sample(n=len(returns), replace=True)
            bootstrap_returns.index = returns.index  # Keep original dates
            
            # Run strategy on bootstrap sample
            try:
                result = strategy_function(bootstrap_returns, parameters)
                self.bootstrap_results.append({
                    'simulation': i,
                    'results': result
                })
                
            except Exception:
                continue
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.n_simulations} simulations")
        
        return self._analyze_bootstrap_results()
    
    def _analyze_bootstrap_results(self) -> Dict[str, Any]:
        """Analyze bootstrap simulation results"""
        if not self.bootstrap_results:
            return {}
        
        # Extract metrics
        returns = []
        sharpe_ratios = []
        
        for sim_result in self.bootstrap_results:
            results = sim_result['results']
            
            if isinstance(results, dict):
                returns.append(results.get('total_return', 0))
                sharpe_ratios.append(results.get('sharpe_ratio', 0))
        
        return {
            'n_simulations': len(self.bootstrap_results),
            'return_mean': np.mean(returns) if returns else 0,
            'return_std': np.std(returns) if returns else 0,
            'sharpe_mean': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'return_confidence_95': (np.percentile(returns, 2.5), np.percentile(returns, 97.5)) if returns else (0, 0)
        }


class ScenarioAnalyzer:
    """Analyze strategy performance under different market scenarios"""
    
    def __init__(self):
        self.scenarios = {}
    
    def add_scenario(self, name: str, data_modifications: Dict[str, Any]):
        """Add a market scenario for analysis"""
        self.scenarios[name] = data_modifications
    
    def analyze_scenarios(self, 
                         strategy_function: Callable,
                         base_data: pd.DataFrame,
                         parameters: Dict[str, Any]) -> pd.DataFrame:
        """Analyze strategy performance across scenarios"""
        scenario_results = {}
        
        # Add baseline scenario
        baseline_result = strategy_function(base_data, parameters)
        scenario_results['Baseline'] = baseline_result
        
        # Test each scenario
        for scenario_name, modifications in self.scenarios.items():
            modified_data = self._modify_data(base_data, modifications)
            result = strategy_function(modified_data, parameters)
            scenario_results[scenario_name] = result
        
        # Create comparison DataFrame
        comparison_data = {}
        for scenario, results in scenario_results.items():
            if isinstance(results, dict):
                comparison_data[scenario] = {
                    'Total Return': results.get('total_return', 0),
                    'Sharpe Ratio': results.get('sharpe_ratio', 0),
                    'Max Drawdown': results.get('max_drawdown', 0)
                }
        
        return pd.DataFrame(comparison_data).T
    
    def _modify_data(self, data: pd.DataFrame, modifications: Dict[str, Any]) -> pd.DataFrame:
        """Apply modifications to create scenario data"""
        modified_data = data.copy()
        
        if 'trend_bias' in modifications:
            bias = modifications['trend_bias']
            if 'Close' in modified_data.columns:
                # Add trend bias to returns
                returns = modified_data['Close'].pct_change()
                modified_returns = returns + bias / 252  # Daily bias
                
                # Reconstruct prices
                for i in range(1, len(modified_data)):
                    prev_price = modified_data.iloc[i-1]['Close']
                    modified_data.iloc[i, modified_data.columns.get_loc('Close')] = prev_price * (1 + modified_returns.iloc[i])
        
        return modified_data 