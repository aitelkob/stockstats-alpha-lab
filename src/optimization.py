"""
Portfolio Optimization and Parameter Tuning Module

This module provides advanced portfolio optimization techniques including
Markowitz optimization, risk parity, Black-Litterman, and parameter tuning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
from concurrent.futures import ThreadPoolExecutor
import itertools

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Advanced portfolio optimization engine."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.optimization_results = {}
    
    def mean_variance_optimization(
        self, 
        returns: pd.DataFrame, 
        target_return: float = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> Dict:
        """
        Mean-variance optimization (Markowitz).
        
        Args:
            returns: DataFrame of asset returns
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(returns.columns)
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        
        if target_return is None:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
                return -(portfolio_return - self.risk_free_rate) / portfolio_std
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        else:
            # Minimize variance for target return
            def objective(weights):
                return weights.T @ cov_matrix @ weights
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
            ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_std = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'success': True,
                'method': 'mean_variance'
            }
        else:
            return {'success': False, 'message': result.message}
    
    def risk_parity_optimization(
        self, 
        returns: pd.DataFrame,
        target_risk_contrib: float = None
    ) -> Dict:
        """
        Risk parity optimization.
        
        Args:
            returns: DataFrame of asset returns
            target_risk_contrib: Target risk contribution per asset (if None, equal risk)
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov()
        
        if target_risk_contrib is None:
            target_risk_contrib = 1.0 / n_assets
        
        def risk_parity_objective(weights):
            portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contrib = (weights * (cov_matrix @ weights)) / (portfolio_std ** 2)
            return np.sum((risk_contrib - target_risk_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(optimal_weights * returns.mean())
            portfolio_std = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'success': True,
                'method': 'risk_parity'
            }
        else:
            return {'success': False, 'message': result.message}
    
    def black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series = None,
        views: Dict[str, float] = None,
        confidence: float = 0.25
    ) -> Dict:
        """
        Black-Litterman optimization with views.
        
        Args:
            returns: DataFrame of asset returns
            market_caps: Market capitalizations for market portfolio
            views: Dictionary of asset views (asset_name: expected_return)
            confidence: Confidence level for views
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(returns.columns)
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Market portfolio weights (equal weight if market_caps not provided)
        if market_caps is None:
            market_weights = np.array([1/n_assets] * n_assets)
        else:
            market_weights = market_caps.values / market_caps.sum()
        
        # Implied equilibrium returns
        risk_aversion = 3.0  # Typical value
        pi = risk_aversion * cov_matrix @ market_weights
        
        if views is None or len(views) == 0:
            # No views, return market portfolio
            portfolio_return = np.sum(market_weights * expected_returns)
            portfolio_std = np.sqrt(market_weights.T @ cov_matrix @ market_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': market_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'success': True,
                'method': 'black_litterman'
            }
        
        # Create view matrix and vector
        view_assets = list(views.keys())
        view_returns = np.array(list(views.values()))
        
        P = np.zeros((len(view_assets), n_assets))
        for i, asset in enumerate(view_assets):
            if asset in returns.columns:
                asset_idx = returns.columns.get_loc(asset)
                P[i, asset_idx] = 1
        
        # Uncertainty matrix
        tau = 0.05  # Typical value
        omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T)) * confidence
        
        # Black-Litterman formula
        M1 = np.linalg.inv(tau * cov_matrix)
        M2 = P.T @ np.linalg.inv(omega) @ P
        M3 = P.T @ np.linalg.inv(omega) @ view_returns
        
        new_cov = np.linalg.inv(M1 + M2)
        new_returns = new_cov @ (M1 @ pi + M3)
        
        # Optimize with new expected returns
        def objective(weights):
            portfolio_return = np.sum(weights * new_returns)
            portfolio_std = np.sqrt(weights.T @ new_cov @ weights)
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = market_weights
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(optimal_weights * new_returns)
            portfolio_std = np.sqrt(optimal_weights.T @ new_cov @ optimal_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'success': True,
                'method': 'black_litterman'
            }
        else:
            return {'success': False, 'message': result.message}
    
    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        linkage_method: str = 'single'
    ) -> Dict:
        """
        Hierarchical Risk Parity optimization.
        
        Args:
            returns: DataFrame of asset returns
            linkage_method: Linkage method for clustering
            
        Returns:
            Dictionary with optimization results
        """
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Convert to distance matrix
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method=linkage_method)
        
        # Build tree and calculate weights
        tree = to_tree(linkage_matrix)
        
        def calculate_hrp_weights(node, cov_matrix):
            if node.is_leaf():
                return np.array([1.0])
            
            left_weights = calculate_hrp_weights(node.left, cov_matrix)
            right_weights = calculate_hrp_weights(node.right, cov_matrix)
            
            # Combine weights based on inverse variance
            left_var = left_weights.T @ cov_matrix.iloc[node.left.pre_order(), node.left.pre_order()] @ left_weights
            right_var = right_weights.T @ cov_matrix.iloc[node.right.pre_order(), node.right.pre_order()] @ right_weights
            
            alpha = 1 - left_var / (left_var + right_var)
            
            combined_weights = np.zeros(len(cov_matrix))
            combined_weights[node.left.pre_order()] = alpha * left_weights
            combined_weights[node.right.pre_order()] = (1 - alpha) * right_weights
            
            return combined_weights
        
        try:
            optimal_weights = calculate_hrp_weights(tree, returns.cov())
            portfolio_return = np.sum(optimal_weights * returns.mean())
            portfolio_std = np.sqrt(optimal_weights.T @ returns.cov() @ optimal_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'success': True,
                'method': 'hierarchical_risk_parity'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}


class ParameterOptimizer:
    """Parameter optimization for trading strategies."""
    
    def __init__(self, n_jobs: int = -1):
        """
        Initialize parameter optimizer.
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_jobs = n_jobs
        self.optimization_results = {}
    
    def optimize_strategy_parameters(
        self,
        strategy_func: Callable,
        parameter_space: Dict[str, Tuple[float, float]],
        data: pd.DataFrame,
        objective_func: Callable,
        method: str = "differential_evolution",
        max_iterations: int = 100
    ) -> Dict:
        """
        Optimize strategy parameters using various methods.
        
        Args:
            strategy_func: Strategy function to optimize
            parameter_space: Dictionary of parameter bounds {param_name: (min, max)}
            data: Market data for backtesting
            objective_func: Objective function to maximize
            method: Optimization method ('differential_evolution', 'grid_search', 'random_search')
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with optimization results
        """
        param_names = list(parameter_space.keys())
        param_bounds = list(parameter_space.values())
        
        def objective_wrapper(params):
            param_dict = dict(zip(param_names, params))
            try:
                result = strategy_func(data, **param_dict)
                return -objective_func(result)  # Minimize negative of objective
            except Exception as e:
                logger.warning(f"Strategy evaluation failed: {e}")
                return 1e6  # Large penalty for failed evaluations
        
        if method == "differential_evolution":
            result = differential_evolution(
                objective_wrapper,
                param_bounds,
                maxiter=max_iterations,
                workers=self.n_jobs,
                seed=42
            )
            
            optimal_params = dict(zip(param_names, result.x))
            best_score = -result.fun
            
        elif method == "grid_search":
            # Generate parameter grid
            param_values = {}
            for param, (min_val, max_val) in parameter_space.items():
                param_values[param] = np.linspace(min_val, max_val, 10)
            
            # Grid search
            best_score = -np.inf
            optimal_params = None
            
            for param_combination in itertools.product(*param_values.values()):
                param_dict = dict(zip(param_names, param_combination))
                try:
                    result = strategy_func(data, **param_dict)
                    score = objective_func(result)
                    if score > best_score:
                        best_score = score
                        optimal_params = param_dict
                except Exception as e:
                    continue
            
            if optimal_params is None:
                return {'success': False, 'message': 'No valid parameter combination found'}
        
        elif method == "random_search":
            best_score = -np.inf
            optimal_params = None
            
            for _ in range(max_iterations):
                param_dict = {}
                for param, (min_val, max_val) in parameter_space.items():
                    param_dict[param] = np.random.uniform(min_val, max_val)
                
                try:
                    result = strategy_func(data, **param_dict)
                    score = objective_func(result)
                    if score > best_score:
                        best_score = score
                        optimal_params = param_dict
                except Exception as e:
                    continue
            
            if optimal_params is None:
                return {'success': False, 'message': 'No valid parameter combination found'}
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return {
            'optimal_parameters': optimal_params,
            'best_score': best_score,
            'success': True,
            'method': method
        }
    
    def walk_forward_optimization(
        self,
        strategy_func: Callable,
        parameter_space: Dict[str, Tuple[float, float]],
        data: pd.DataFrame,
        objective_func: Callable,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21
    ) -> Dict:
        """
        Walk-forward parameter optimization.
        
        Args:
            strategy_func: Strategy function to optimize
            parameter_space: Dictionary of parameter bounds
            data: Market data
            objective_func: Objective function
            train_window: Training window size (days)
            test_window: Test window size (days)
            step_size: Step size for rolling window
            
        Returns:
            Dictionary with walk-forward results
        """
        results = []
        current_start = 0
        
        while current_start + train_window + test_window <= len(data):
            # Training data
            train_data = data.iloc[current_start:current_start + train_window]
            
            # Test data
            test_start = current_start + train_window
            test_end = test_start + test_window
            test_data = data.iloc[test_start:test_end]
            
            # Optimize on training data
            opt_result = self.optimize_strategy_parameters(
                strategy_func, parameter_space, train_data, objective_func
            )
            
            if opt_result['success']:
                # Test on out-of-sample data
                try:
                    test_result = strategy_func(test_data, **opt_result['optimal_parameters'])
                    test_score = objective_func(test_result)
                    
                    results.append({
                        'train_start': current_start,
                        'train_end': current_start + train_window,
                        'test_start': test_start,
                        'test_end': test_end,
                        'optimal_parameters': opt_result['optimal_parameters'],
                        'train_score': opt_result['best_score'],
                        'test_score': test_score,
                        'overfitting': opt_result['best_score'] - test_score
                    })
                except Exception as e:
                    logger.warning(f"Test evaluation failed: {e}")
            
            current_start += step_size
        
        if not results:
            return {'success': False, 'message': 'No valid walk-forward results'}
        
        # Calculate summary statistics
        test_scores = [r['test_score'] for r in results]
        overfitting = [r['overfitting'] for r in results]
        
        return {
            'results': results,
            'mean_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'mean_overfitting': np.mean(overfitting),
            'success': True,
            'method': 'walk_forward'
        }


class MultiObjectiveOptimizer:
    """Multi-objective optimization for portfolio strategies."""
    
    def __init__(self):
        """Initialize multi-objective optimizer."""
        self.pareto_front = []
    
    def optimize_multi_objective(
        self,
        strategy_func: Callable,
        parameter_space: Dict[str, Tuple[float, float]],
        data: pd.DataFrame,
        objectives: List[Callable],
        weights: List[float] = None,
        method: str = "weighted_sum"
    ) -> Dict:
        """
        Multi-objective optimization.
        
        Args:
            strategy_func: Strategy function
            parameter_space: Parameter bounds
            objectives: List of objective functions
            weights: Weights for objectives (if None, equal weights)
            method: Optimization method ('weighted_sum', 'pareto')
            
        Returns:
            Dictionary with optimization results
        """
        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)
        
        param_names = list(parameter_space.keys())
        param_bounds = list(parameter_space.values())
        
        def multi_objective_wrapper(params):
            param_dict = dict(zip(param_names, params))
            try:
                result = strategy_func(data, **param_dict)
                objective_values = [obj(result) for obj in objectives]
                
                if method == "weighted_sum":
                    return -np.sum([w * obj_val for w, obj_val in zip(weights, objective_values)])
                elif method == "pareto":
                    return objective_values
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as e:
                logger.warning(f"Strategy evaluation failed: {e}")
                return [1e6] * len(objectives) if method == "pareto" else 1e6
        
        if method == "weighted_sum":
            result = differential_evolution(
                multi_objective_wrapper,
                param_bounds,
                maxiter=100,
                seed=42
            )
            
            optimal_params = dict(zip(param_names, result.x))
            best_score = -result.fun
            
            return {
                'optimal_parameters': optimal_params,
                'best_score': best_score,
                'success': True,
                'method': 'weighted_sum'
            }
        
        elif method == "pareto":
            # Generate Pareto front
            pareto_solutions = []
            
            # Sample parameter space
            n_samples = 1000
            for _ in range(n_samples):
                params = [np.random.uniform(min_val, max_val) for min_val, max_val in param_bounds]
                param_dict = dict(zip(param_names, params))
                
                try:
                    result = strategy_func(data, **param_dict)
                    objective_values = [obj(result) for obj in objectives]
                    pareto_solutions.append((param_dict, objective_values))
                except Exception as e:
                    continue
            
            # Find Pareto optimal solutions
            pareto_optimal = []
            for i, (params1, obj1) in enumerate(pareto_solutions):
                is_pareto = True
                for j, (params2, obj2) in enumerate(pareto_solutions):
                    if i != j:
                        # Check if obj2 dominates obj1
                        if all(o2 >= o1 for o2, o1 in zip(obj2, obj1)) and any(o2 > o1 for o2, o1 in zip(obj2, obj1)):
                            is_pareto = False
                            break
                if is_pareto:
                    pareto_optimal.append((params1, obj1))
            
            self.pareto_front = pareto_optimal
            
            return {
                'pareto_front': pareto_optimal,
                'num_solutions': len(pareto_optimal),
                'success': True,
                'method': 'pareto'
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")


def create_efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 100,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Create efficient frontier for portfolio optimization.
    
    Args:
        returns: DataFrame of asset returns
        n_portfolios: Number of portfolios to generate
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with efficient frontier data
    """
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Generate random portfolios
    n_assets = len(returns.columns)
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe_ratios = []
    portfolio_weights = []
    
    for _ in range(n_portfolios):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)
        portfolio_sharpe_ratios.append(sharpe_ratio)
        portfolio_weights.append(weights)
    
    # Create DataFrame
    efficient_frontier = pd.DataFrame({
        'return': portfolio_returns,
        'volatility': portfolio_volatilities,
        'sharpe_ratio': portfolio_sharpe_ratios
    })
    
    # Add weights for each asset
    for i, asset in enumerate(returns.columns):
        efficient_frontier[f'weight_{asset}'] = [w[i] for w in portfolio_weights]
    
    return efficient_frontier.sort_values('volatility')
