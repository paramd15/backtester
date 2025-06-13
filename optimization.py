"""Optimization Module - Phase 3 Advanced Analytics"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Tuple
from itertools import product
import random


class ParameterOptimizer:
    """Base class for parameter optimization"""
    
    def __init__(self, objective_function: Callable, maximize: bool = True):
        self.objective_function = objective_function
        self.maximize = maximize
        self.optimization_results = []
    
    def optimize(self, parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """Run optimization - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found"""
        if not self.optimization_results:
            return {}
        
        best_result = max(self.optimization_results, 
                         key=lambda x: x['score'] if self.maximize else lambda x: -x['score'])
        return best_result


class GridSearchOptimizer(ParameterOptimizer):
    """Grid Search Parameter Optimization"""
    
    def optimize(self, parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """Perform grid search optimization"""
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        combinations = list(product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            score = self.objective_function(params)
            
            self.optimization_results.append({
                'parameters': params,
                'score': score,
                'iteration': i
            })
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{len(combinations)} combinations")
        
        best_result = self.get_best_parameters()
        print(f"Best score: {best_result['score']:.4f}")
        print(f"Best parameters: {best_result['parameters']}")
        
        return best_result
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get optimization results as DataFrame"""
        if not self.optimization_results:
            return pd.DataFrame()
        
        data = []
        for result in self.optimization_results:
            row = result['parameters'].copy()
            row['score'] = result['score']
            data.append(row)
        
        return pd.DataFrame(data)


class GeneticOptimizer(ParameterOptimizer):
    """Genetic Algorithm Parameter Optimization"""
    
    def __init__(self, 
                 objective_function: Callable,
                 maximize: bool = True,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1):
        super().__init__(objective_function, maximize)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def optimize(self, parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """Perform genetic algorithm optimization"""
        param_names = list(parameter_space.keys())
        param_ranges = {name: (min(values), max(values)) for name, values in parameter_space.items()}
        
        # Initialize population
        population = self._initialize_population(param_names, param_ranges)
        
        print(f"Starting genetic optimization: {self.generations} generations")
        
        for generation in range(self.generations):
            # Evaluate population
            scores = []
            for individual in population:
                params = dict(zip(param_names, individual))
                score = self.objective_function(params)
                scores.append(score)
                
                self.optimization_results.append({
                    'parameters': params,
                    'score': score,
                    'generation': generation
                })
            
            # Evolve population
            population = self._evolve_population(population, scores, param_ranges)
            
            if generation % 20 == 0:
                best_score = max(scores) if self.maximize else min(scores)
                print(f"Generation {generation}: Best score = {best_score:.4f}")
        
        best_result = self.get_best_parameters()
        print(f"Final best score: {best_result['score']:.4f}")
        
        return best_result
    
    def _initialize_population(self, param_names: List[str], param_ranges: Dict) -> List[List]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for param in param_names:
                min_val, max_val = param_ranges[param]
                value = random.uniform(min_val, max_val)
                individual.append(value)
            population.append(individual)
        return population
    
    def _evolve_population(self, population: List[List], scores: List[float], param_ranges: Dict) -> List[List]:
        """Evolve population through selection, crossover, and mutation"""
        # Sort by fitness
        population_with_scores = list(zip(population, scores))
        population_with_scores.sort(key=lambda x: x[1], reverse=self.maximize)
        
        # Keep top 20% as elite
        elite_size = max(1, self.population_size // 5)
        new_population = [ind for ind, _ in population_with_scores[:elite_size]]
        
        # Generate rest through crossover and mutation
        param_names = list(param_ranges.keys())
        
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._tournament_selection(population_with_scores)
            parent2 = self._tournament_selection(population_with_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child, param_ranges, param_names)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population_with_scores: List[Tuple]) -> List:
        """Tournament selection"""
        tournament_size = 3
        tournament = random.sample(population_with_scores, min(tournament_size, len(population_with_scores)))
        winner = max(tournament, key=lambda x: x[1] if self.maximize else lambda x: -x[1])
        return winner[0]
    
    def _crossover(self, parent1: List, parent2: List) -> List:
        """Single-point crossover"""
        if len(parent1) <= 1:
            return parent1.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def _mutate(self, individual: List, param_ranges: Dict, param_names: List[str]) -> List:
        """Gaussian mutation"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                min_val, max_val = param_ranges[param_names[i]]
                # Add small random change
                change = random.gauss(0, (max_val - min_val) * 0.1)
                mutated[i] = max(min_val, min(max_val, mutated[i] + change))
        
        return mutated


class WalkForwardOptimizer:
    """Walk-Forward Analysis and Optimization"""
    
    def __init__(self, 
                 optimization_function: Callable,
                 backtest_function: Callable,
                 train_window: int = 252,
                 test_window: int = 63):
        self.optimization_function = optimization_function
        self.backtest_function = backtest_function
        self.train_window = train_window
        self.test_window = test_window
        self.walk_forward_results = []
    
    def run_walk_forward(self, 
                        data: pd.DataFrame,
                        parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        print("Starting walk-forward analysis...")
        
        total_periods = len(data)
        start_idx = self.train_window
        
        while start_idx + self.test_window < total_periods:
            # Define training and testing periods
            train_data = data.iloc[start_idx - self.train_window:start_idx]
            test_data = data.iloc[start_idx:start_idx + self.test_window]
            
            print(f"Training: {train_data.index[0]} to {train_data.index[-1]}")
            
            # Optimize parameters on training data
            best_params = self.optimization_function(train_data, parameter_space)
            
            # Test on out-of-sample data
            test_results = self.backtest_function(test_data, best_params)
            
            self.walk_forward_results.append({
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'best_parameters': best_params,
                'test_results': test_results
            })
            
            start_idx += self.test_window
        
        return self._analyze_walk_forward_results()
    
    def _analyze_walk_forward_results(self) -> Dict[str, Any]:
        """Analyze walk-forward results"""
        if not self.walk_forward_results:
            return {}
        
        # Extract test performance
        all_returns = []
        for result in self.walk_forward_results:
            test_results = result['test_results']
            if isinstance(test_results, dict) and 'returns' in test_results:
                all_returns.extend(test_results['returns'])
        
        if all_returns:
            returns_series = pd.Series(all_returns)
            total_return = (1 + returns_series).prod() - 1
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() != 0 else 0
        else:
            total_return = sharpe = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'num_periods': len(self.walk_forward_results),
            'period_results': self.walk_forward_results
        }


def create_parameter_space(param_configs: Dict[str, Dict]) -> Dict[str, List]:
    """Create parameter space from configuration"""
    parameter_space = {}
    
    for param_name, config in param_configs.items():
        if 'values' in config:
            parameter_space[param_name] = config['values']
        elif 'min' in config and 'max' in config:
            min_val = config['min']
            max_val = config['max']
            step = config.get('step', (max_val - min_val) / 10)
            
            values = []
            current = min_val
            while current <= max_val:
                values.append(current)
                current += step
            
            parameter_space[param_name] = values
    
    return parameter_space 