"""
Tests for Portfolio Optimization Module

This module tests the portfolio optimization functionality
including mean-variance, risk parity, and parameter optimization.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
import sys
sys.path.append('src')

from optimization import (
    PortfolioOptimizer,
    ParameterOptimizer,
    MultiObjectiveOptimizer,
    create_efficient_frontier
)


class TestPortfolioOptimizer:
    """Test the PortfolioOptimizer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 100),
            'Asset2': np.random.normal(0.0008, 0.018, 100),
            'Asset3': np.random.normal(0.0012, 0.025, 100),
            'Asset4': np.random.normal(0.0009, 0.022, 100)
        }, index=dates)
    
    @pytest.fixture
    def optimizer(self):
        """Create PortfolioOptimizer instance."""
        return PortfolioOptimizer()
    
    def test_mean_variance_optimization_maximize_sharpe(self, optimizer, sample_returns):
        """Test mean-variance optimization maximizing Sharpe ratio."""
        result = optimizer.mean_variance_optimization(sample_returns)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'weights' in result
            assert 'expected_return' in result
            assert 'volatility' in result
            assert 'sharpe_ratio' in result
            assert 'method' in result
            
            # Weights should sum to 1
            assert abs(np.sum(result['weights']) - 1.0) < 1e-6
            
            # Weights should be non-negative
            assert np.all(result['weights'] >= 0)
            
            # Method should be mean_variance
            assert result['method'] == 'mean_variance'
    
    def test_mean_variance_optimization_target_return(self, optimizer, sample_returns):
        """Test mean-variance optimization with target return."""
        target_return = 0.001
        result = optimizer.mean_variance_optimization(sample_returns, target_return=target_return)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # Check that target return is achieved (approximately)
            assert abs(result['expected_return'] - target_return) < 1e-3
    
    def test_mean_variance_optimization_with_constraints(self, optimizer, sample_returns):
        """Test mean-variance optimization with weight constraints."""
        result = optimizer.mean_variance_optimization(
            sample_returns, 
            max_weight=0.5, 
            min_weight=0.1
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # Check weight constraints
            assert np.all(result['weights'] >= 0.1)
            assert np.all(result['weights'] <= 0.5)
    
    def test_risk_parity_optimization(self, optimizer, sample_returns):
        """Test risk parity optimization."""
        result = optimizer.risk_parity_optimization(sample_returns)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'weights' in result
            assert 'expected_return' in result
            assert 'volatility' in result
            assert 'sharpe_ratio' in result
            assert 'method' in result
            
            # Weights should sum to 1
            assert abs(np.sum(result['weights']) - 1.0) < 1e-6
            
            # Weights should be non-negative
            assert np.all(result['weights'] >= 0)
            
            # Method should be risk_parity
            assert result['method'] == 'risk_parity'
    
    def test_risk_parity_optimization_custom_target(self, optimizer, sample_returns):
        """Test risk parity optimization with custom target risk contribution."""
        target_risk_contrib = 0.3
        result = optimizer.risk_parity_optimization(sample_returns, target_risk_contrib)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # Weights should sum to 1
            assert abs(np.sum(result['weights']) - 1.0) < 1e-6
    
    def test_black_litterman_optimization_no_views(self, optimizer, sample_returns):
        """Test Black-Litterman optimization without views."""
        result = optimizer.black_litterman_optimization(sample_returns)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'weights' in result
            assert 'expected_return' in result
            assert 'volatility' in result
            assert 'sharpe_ratio' in result
            assert 'method' in result
            
            # Weights should sum to 1
            assert abs(np.sum(result['weights']) - 1.0) < 1e-6
            
            # Method should be black_litterman
            assert result['method'] == 'black_litterman'
    
    def test_black_litterman_optimization_with_views(self, optimizer, sample_returns):
        """Test Black-Litterman optimization with views."""
        views = {'Asset1': 0.002, 'Asset2': 0.001}  # Expected returns for some assets
        result = optimizer.black_litterman_optimization(sample_returns, views=views)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'weights' in result
            assert 'expected_return' in result
            assert 'volatility' in result
            assert 'sharpe_ratio' in result
    
    def test_black_litterman_optimization_with_market_caps(self, optimizer, sample_returns):
        """Test Black-Litterman optimization with market capitalizations."""
        market_caps = pd.Series([1000, 2000, 1500, 1200], index=sample_returns.columns)
        views = {'Asset1': 0.002}
        
        result = optimizer.black_litterman_optimization(
            sample_returns, 
            market_caps=market_caps, 
            views=views
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_hierarchical_risk_parity(self, optimizer, sample_returns):
        """Test hierarchical risk parity optimization."""
        result = optimizer.hierarchical_risk_parity(sample_returns)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'weights' in result
            assert 'expected_return' in result
            assert 'volatility' in result
            assert 'sharpe_ratio' in result
            assert 'method' in result
            
            # Weights should sum to 1
            assert abs(np.sum(result['weights']) - 1.0) < 1e-6
            
            # Method should be hierarchical_risk_parity
            assert result['method'] == 'hierarchical_risk_parity'


class TestParameterOptimizer:
    """Test the ParameterOptimizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100)),
            'rsi_14': np.random.uniform(20, 80, 100),
            'close_20_ema': 100 * np.cumprod(1 + np.random.normal(0, 0.015, 100))
        }, index=dates)
    
    @pytest.fixture
    def strategy_function(self):
        """Create a simple strategy function for testing."""
        def strategy_func(data, param1=30, param2=70):
            """Simple strategy function for testing."""
            if 'rsi_14' not in data.columns:
                return {'total_return': 0, 'sharpe_ratio': 0}
            
            # Simple strategy based on RSI
            signals = (data['rsi_14'] < param1).astype(int) - (data['rsi_14'] > param2).astype(int)
            returns = signals * data['close'].pct_change()
            total_return = returns.sum()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            return {'total_return': total_return, 'sharpe_ratio': sharpe_ratio}
        
        return strategy_func
    
    @pytest.fixture
    def objective_function(self):
        """Create objective function for testing."""
        def objective_func(result):
            return result['sharpe_ratio']
        return objective_func
    
    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        return {
            'param1': (20, 40),
            'param2': (60, 80)
        }
    
    @pytest.fixture
    def param_optimizer(self):
        """Create ParameterOptimizer instance."""
        return ParameterOptimizer()
    
    def test_optimize_strategy_parameters_differential_evolution(self, param_optimizer, strategy_function, parameter_space, sample_data, objective_function):
        """Test parameter optimization using differential evolution."""
        result = param_optimizer.optimize_strategy_parameters(
            strategy_function,
            parameter_space,
            sample_data,
            objective_function,
            method='differential_evolution',
            max_iterations=10  # Small number for testing
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'optimal_parameters' in result
            assert 'best_score' in result
            assert 'method' in result
            
            # Check that optimal parameters are within bounds
            for param, value in result['optimal_parameters'].items():
                min_val, max_val = parameter_space[param]
                assert min_val <= value <= max_val
            
            assert result['method'] == 'differential_evolution'
    
    def test_optimize_strategy_parameters_grid_search(self, param_optimizer, strategy_function, parameter_space, sample_data, objective_function):
        """Test parameter optimization using grid search."""
        result = param_optimizer.optimize_strategy_parameters(
            strategy_function,
            parameter_space,
            sample_data,
            objective_function,
            method='grid_search',
            max_iterations=10
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'optimal_parameters' in result
            assert 'best_score' in result
            assert 'method' in result
            assert result['method'] == 'grid_search'
    
    def test_optimize_strategy_parameters_random_search(self, param_optimizer, strategy_function, parameter_space, sample_data, objective_function):
        """Test parameter optimization using random search."""
        result = param_optimizer.optimize_strategy_parameters(
            strategy_function,
            parameter_space,
            sample_data,
            objective_function,
            method='random_search',
            max_iterations=10
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'optimal_parameters' in result
            assert 'best_score' in result
            assert 'method' in result
            assert result['method'] == 'random_search'
    
    def test_walk_forward_optimization(self, param_optimizer, strategy_function, parameter_space, sample_data, objective_function):
        """Test walk-forward parameter optimization."""
        result = param_optimizer.walk_forward_optimization(
            strategy_function,
            parameter_space,
            sample_data,
            objective_function,
            train_window=50,
            test_window=20,
            step_size=10
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'results' in result
            assert 'mean_test_score' in result
            assert 'std_test_score' in result
            assert 'mean_overfitting' in result
            assert 'method' in result
            
            assert result['method'] == 'walk_forward'
            assert len(result['results']) > 0


class TestMultiObjectiveOptimizer:
    """Test the MultiObjectiveOptimizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100)),
            'rsi_14': np.random.uniform(20, 80, 100),
            'close_20_ema': 100 * np.cumprod(1 + np.random.normal(0, 0.015, 100))
        }, index=dates)
    
    @pytest.fixture
    def strategy_function(self):
        """Create a simple strategy function for testing."""
        def strategy_func(data, param1=30, param2=70):
            """Simple strategy function for testing."""
            if 'rsi_14' not in data.columns:
                return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
            
            # Simple strategy based on RSI
            signals = (data['rsi_14'] < param1).astype(int) - (data['rsi_14'] > param2).astype(int)
            returns = signals * data['close'].pct_change()
            total_return = returns.sum()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = returns.cumsum().expanding().max() - returns.cumsum()
            max_drawdown = max_drawdown.max() if len(max_drawdown) > 0 else 0
            
            return {'total_return': total_return, 'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown}
        
        return strategy_func
    
    @pytest.fixture
    def objectives(self):
        """Create objective functions for testing."""
        def objective1(result):
            return result['sharpe_ratio']
        
        def objective2(result):
            return -result['max_drawdown']  # Minimize drawdown
        
        return [objective1, objective2]
    
    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        return {
            'param1': (20, 40),
            'param2': (60, 80)
        }
    
    @pytest.fixture
    def multi_optimizer(self):
        """Create MultiObjectiveOptimizer instance."""
        return MultiObjectiveOptimizer()
    
    def test_optimize_multi_objective_weighted_sum(self, multi_optimizer, strategy_function, parameter_space, sample_data, objectives):
        """Test multi-objective optimization using weighted sum."""
        weights = [0.7, 0.3]
        result = multi_optimizer.optimize_multi_objective(
            strategy_function,
            parameter_space,
            sample_data,
            objectives,
            weights=weights,
            method='weighted_sum'
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'optimal_parameters' in result
            assert 'best_score' in result
            assert 'method' in result
            assert result['method'] == 'weighted_sum'
    
    def test_optimize_multi_objective_pareto(self, multi_optimizer, strategy_function, parameter_space, sample_data, objectives):
        """Test multi-objective optimization using Pareto frontier."""
        result = multi_optimizer.optimize_multi_objective(
            strategy_function,
            parameter_space,
            sample_data,
            objectives,
            method='pareto'
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'pareto_front' in result
            assert 'num_solutions' in result
            assert 'method' in result
            assert result['method'] == 'pareto'
            assert result['num_solutions'] > 0


class TestEfficientFrontier:
    """Test efficient frontier creation."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 100),
            'Asset2': np.random.normal(0.0008, 0.018, 100),
            'Asset3': np.random.normal(0.0012, 0.025, 100)
        }, index=dates)
    
    def test_create_efficient_frontier(self, sample_returns):
        """Test efficient frontier creation."""
        frontier = create_efficient_frontier(sample_returns, n_portfolios=50)
        
        assert isinstance(frontier, pd.DataFrame)
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns
        assert 'sharpe_ratio' in frontier.columns
        
        # Check that we have the expected number of portfolios
        assert len(frontier) == 50
        
        # Check that returns and volatilities are reasonable
        assert frontier['return'].min() < frontier['return'].max()
        assert frontier['volatility'].min() < frontier['volatility'].max()
        
        # Check that weights sum to 1 for each portfolio
        weight_cols = [col for col in frontier.columns if col.startswith('weight_')]
        for _, row in frontier.iterrows():
            weight_sum = sum(row[col] for col in weight_cols)
            assert abs(weight_sum - 1.0) < 1e-6


class TestOptimizationEdgeCases:
    """Test edge cases for optimization."""
    
    def test_optimization_with_insufficient_data(self):
        """Test optimization with insufficient data."""
        optimizer = PortfolioOptimizer()
        
        # Create returns with only 2 observations
        returns = pd.DataFrame({
            'Asset1': [0.01, -0.02],
            'Asset2': [0.015, -0.01]
        })
        
        result = optimizer.mean_variance_optimization(returns)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_optimization_with_constant_returns(self):
        """Test optimization with constant returns."""
        optimizer = PortfolioOptimizer()
        
        # Create returns with constant values
        returns = pd.DataFrame({
            'Asset1': [0.01] * 10,
            'Asset2': [0.01] * 10,
            'Asset3': [0.01] * 10
        })
        
        result = optimizer.mean_variance_optimization(returns)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_optimization_with_negative_returns(self):
        """Test optimization with all negative returns."""
        optimizer = PortfolioOptimizer()
        
        # Create returns with all negative values
        returns = pd.DataFrame({
            'Asset1': [-0.01] * 10,
            'Asset2': [-0.02] * 10,
            'Asset3': [-0.015] * 10
        })
        
        result = optimizer.mean_variance_optimization(returns)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result
