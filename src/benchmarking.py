"""
Benchmarking and Performance Attribution Module

This module provides comprehensive benchmarking tools, performance attribution
analysis, and factor decomposition for quantitative finance strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """Comprehensive benchmark analysis and performance attribution."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize benchmark analyzer.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_data = {}
        self.attribution_results = {}
    
    def add_benchmark(
        self, 
        name: str, 
        returns: pd.Series, 
        description: str = None
    ) -> None:
        """
        Add a benchmark for comparison.
        
        Args:
            name: Benchmark name
            returns: Benchmark returns series
            description: Optional description
        """
        self.benchmark_data[name] = {
            'returns': returns,
            'description': description or name
        }
        logger.info(f"Added benchmark: {name}")
    
    def calculate_relative_performance(
        self, 
        strategy_returns: pd.Series, 
        benchmark_name: str = None
    ) -> Dict[str, float]:
        """
        Calculate relative performance metrics.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_name: Benchmark to compare against (if None, use first benchmark)
            
        Returns:
            Dictionary with relative performance metrics
        """
        if not self.benchmark_data:
            raise ValueError("No benchmarks available")
        
        if benchmark_name is None:
            benchmark_name = list(self.benchmark_data.keys())[0]
        
        if benchmark_name not in self.benchmark_data:
            raise ValueError(f"Benchmark {benchmark_name} not found")
        
        benchmark_returns = self.benchmark_data[benchmark_name]['returns']
        
        # Align dates
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate excess returns
        excess_returns = strategy_aligned - benchmark_aligned
        
        # Basic metrics
        total_excess_return = (1 + excess_returns).prod() - 1
        annualized_excess_return = (1 + total_excess_return) ** (252 / len(excess_returns)) - 1
        
        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = annualized_excess_return / tracking_error if tracking_error > 0 else 0
        
        # Beta and alpha
        beta, alpha, r_squared, _, _ = stats.linregress(benchmark_aligned, strategy_aligned)
        alpha_annualized = alpha * 252
        
        # Up and down capture
        up_periods = benchmark_aligned > 0
        down_periods = benchmark_aligned < 0
        
        up_capture = (strategy_aligned[up_periods].mean() / benchmark_aligned[up_periods].mean()) if up_periods.any() else 0
        down_capture = (strategy_aligned[down_periods].mean() / benchmark_aligned[down_periods].mean()) if down_periods.any() else 0
        
        # Win rate
        win_rate = (excess_returns > 0).mean()
        
        # Maximum relative drawdown
        cumulative_excess = (1 + excess_returns).cumprod()
        running_max = cumulative_excess.expanding().max()
        relative_drawdown = (cumulative_excess - running_max) / running_max
        max_relative_drawdown = relative_drawdown.min()
        
        return {
            'benchmark': benchmark_name,
            'total_excess_return': total_excess_return,
            'annualized_excess_return': annualized_excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'alpha_annualized': alpha_annualized,
            'r_squared': r_squared,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'win_rate': win_rate,
            'max_relative_drawdown': max_relative_drawdown
        }
    
    def calculate_rolling_attribution(
        self, 
        strategy_returns: pd.Series, 
        benchmark_name: str, 
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance attribution.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_name: Benchmark name
            window: Rolling window size
            
        Returns:
            DataFrame with rolling attribution metrics
        """
        if benchmark_name not in self.benchmark_data:
            raise ValueError(f"Benchmark {benchmark_name} not found")
        
        benchmark_returns = self.benchmark_data[benchmark_name]['returns']
        
        # Align dates
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate rolling metrics
        rolling_alpha = []
        rolling_beta = []
        rolling_r_squared = []
        rolling_tracking_error = []
        rolling_information_ratio = []
        
        for i in range(window, len(strategy_aligned)):
            strategy_window = strategy_aligned.iloc[i-window:i]
            benchmark_window = benchmark_aligned.iloc[i-window:i]
            
            # Regression
            beta, alpha, r_squared, _, _ = stats.linregress(benchmark_window, strategy_window)
            
            # Tracking error
            excess_returns = strategy_window - benchmark_window
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Information ratio
            annualized_excess = (strategy_window.mean() - benchmark_window.mean()) * 252
            information_ratio = annualized_excess / tracking_error if tracking_error > 0 else 0
            
            rolling_alpha.append(alpha * 252)  # Annualized
            rolling_beta.append(beta)
            rolling_r_squared.append(r_squared)
            rolling_tracking_error.append(tracking_error)
            rolling_information_ratio.append(information_ratio)
        
        # Create DataFrame
        attribution_df = pd.DataFrame({
            'date': strategy_aligned.index[window:],
            'alpha': rolling_alpha,
            'beta': rolling_beta,
            'r_squared': rolling_r_squared,
            'tracking_error': rolling_tracking_error,
            'information_ratio': rolling_information_ratio
        })
        
        attribution_df.set_index('date', inplace=True)
        return attribution_df
    
    def factor_attribution(
        self, 
        strategy_returns: pd.Series, 
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Factor-based performance attribution.
        
        Args:
            strategy_returns: Strategy returns
            factor_returns: DataFrame of factor returns
            
        Returns:
            Dictionary with factor attribution results
        """
        # Align dates
        common_dates = strategy_returns.index.intersection(factor_returns.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        # Run regression
        model = LinearRegression()
        model.fit(factors_aligned, strategy_aligned)
        
        # Calculate factor exposures
        factor_exposures = dict(zip(factors_aligned.columns, model.coef_))
        
        # Calculate factor contributions
        factor_contributions = {}
        for factor, exposure in factor_exposures.items():
            factor_contributions[factor] = exposure * factors_aligned[factor].mean() * 252
        
        # Calculate R-squared
        r_squared = model.score(factors_aligned, strategy_aligned)
        
        # Calculate residual return
        predicted_returns = model.predict(factors_aligned)
        residual_returns = strategy_aligned - predicted_returns
        residual_volatility = residual_returns.std() * np.sqrt(252)
        
        return {
            'factor_exposures': factor_exposures,
            'factor_contributions': factor_contributions,
            'r_squared': r_squared,
            'residual_volatility': residual_volatility,
            'alpha': model.intercept_ * 252,  # Annualized
            'total_explained_variance': r_squared
        }
    
    def style_analysis(
        self, 
        strategy_returns: pd.Series, 
        style_factors: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Style analysis using constrained regression.
        
        Args:
            strategy_returns: Strategy returns
            style_factors: DataFrame of style factor returns
            
        Returns:
            Dictionary with style analysis results
        """
        from scipy.optimize import minimize
        
        # Align dates
        common_dates = strategy_returns.index.intersection(style_factors.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        factors_aligned = style_factors.loc[common_dates]
        
        n_factors = len(factors_aligned.columns)
        
        # Objective function (minimize tracking error)
        def objective(weights):
            predicted_returns = factors_aligned @ weights
            tracking_error = np.sum((strategy_aligned - predicted_returns) ** 2)
            return tracking_error
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n_factors)]
        
        # Initial guess
        x0 = np.array([1/n_factors] * n_factors)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            style_weights = dict(zip(factors_aligned.columns, result.x))
            
            # Calculate R-squared
            predicted_returns = factors_aligned @ result.x
            r_squared = 1 - np.sum((strategy_aligned - predicted_returns) ** 2) / np.sum((strategy_aligned - strategy_aligned.mean()) ** 2)
            
            # Calculate tracking error
            tracking_error = np.sqrt(np.mean((strategy_aligned - predicted_returns) ** 2)) * np.sqrt(252)
            
            return {
                'style_weights': style_weights,
                'r_squared': r_squared,
                'tracking_error': tracking_error,
                'success': True
            }
        else:
            return {
                'style_weights': None,
                'r_squared': 0,
                'tracking_error': np.inf,
                'success': False,
                'message': result.message
            }
    
    def calculate_attribution_metrics(
        self, 
        strategy_returns: pd.Series, 
        benchmark_name: str = None
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Calculate comprehensive attribution metrics.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_name: Benchmark name
            
        Returns:
            Dictionary with all attribution metrics
        """
        if not self.benchmark_data:
            raise ValueError("No benchmarks available")
        
        if benchmark_name is None:
            benchmark_name = list(self.benchmark_data.keys())[0]
        
        # Basic relative performance
        relative_performance = self.calculate_relative_performance(strategy_returns, benchmark_name)
        
        # Rolling attribution
        rolling_attribution = self.calculate_rolling_attribution(strategy_returns, benchmark_name)
        
        # Store results
        self.attribution_results[benchmark_name] = {
            'relative_performance': relative_performance,
            'rolling_attribution': rolling_attribution
        }
        
        return {
            'relative_performance': relative_performance,
            'rolling_attribution': rolling_attribution
        }


class PerformanceAttributor:
    """Advanced performance attribution analysis."""
    
    def __init__(self):
        """Initialize performance attributor."""
        self.attribution_results = {}
    
    def brinson_attribution(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series,
        sector_returns: pd.DataFrame,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series
    ) -> Dict[str, float]:
        """
        Brinson attribution analysis.
        
        Args:
            portfolio_returns: Portfolio returns by sector
            benchmark_returns: Benchmark returns by sector
            sector_returns: Market sector returns
            portfolio_weights: Portfolio sector weights
            benchmark_weights: Benchmark sector weights
            
        Returns:
            Dictionary with Brinson attribution results
        """
        # Align all data
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate attribution effects
        # Allocation effect: (w_p - w_b) * (R_s - R_b)
        allocation_effect = (portfolio_weights - benchmark_weights) * (sector_returns.mean() - benchmark_aligned.mean())
        
        # Selection effect: w_b * (R_p - R_s)
        selection_effect = benchmark_weights * (portfolio_aligned.mean() - sector_returns.mean())
        
        # Interaction effect: (w_p - w_b) * (R_p - R_s)
        interaction_effect = (portfolio_weights - benchmark_weights) * (portfolio_aligned.mean() - sector_returns.mean())
        
        # Total attribution
        total_attribution = allocation_effect.sum() + selection_effect.sum() + interaction_effect.sum()
        
        return {
            'allocation_effect': allocation_effect.sum(),
            'selection_effect': selection_effect.sum(),
            'interaction_effect': interaction_effect.sum(),
            'total_attribution': total_attribution,
            'allocation_by_sector': allocation_effect.to_dict(),
            'selection_by_sector': selection_effect.to_dict(),
            'interaction_by_sector': interaction_effect.to_dict()
        }
    
    def factor_attribution_decomposition(
        self, 
        strategy_returns: pd.Series, 
        factor_returns: pd.DataFrame,
        factor_loadings: pd.DataFrame = None
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Factor attribution decomposition.
        
        Args:
            strategy_returns: Strategy returns
            factor_returns: Factor returns
            factor_loadings: Factor loadings (if None, estimate via regression)
            
        Returns:
            Dictionary with factor attribution decomposition
        """
        # Align dates
        common_dates = strategy_returns.index.intersection(factor_returns.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        if factor_loadings is None:
            # Estimate factor loadings via regression
            model = LinearRegression()
            model.fit(factors_aligned, strategy_aligned)
            factor_loadings = pd.Series(model.coef_, index=factors_aligned.columns)
            alpha = model.intercept_
        else:
            # Use provided factor loadings
            alpha = 0  # Assume alpha is included in loadings
        
        # Calculate factor contributions
        factor_contributions = {}
        for factor in factors_aligned.columns:
            factor_contributions[factor] = factor_loadings[factor] * factors_aligned[factor]
        
        # Calculate total explained return
        explained_return = sum(factor_contributions.values())
        residual_return = strategy_aligned - explained_return - alpha
        
        # Calculate attribution metrics
        total_variance = strategy_aligned.var()
        explained_variance = explained_return.var()
        residual_variance = residual_return.var()
        
        r_squared = explained_variance / total_variance if total_variance > 0 else 0
        
        # Factor risk contributions
        factor_risk_contributions = {}
        for factor in factors_aligned.columns:
            factor_risk_contributions[factor] = factor_loadings[factor] * factors_aligned[factor].std()
        
        return {
            'factor_loadings': factor_loadings,
            'factor_contributions': factor_contributions,
            'alpha': alpha,
            'residual_return': residual_return,
            'r_squared': r_squared,
            'explained_variance': explained_variance,
            'residual_variance': residual_variance,
            'factor_risk_contributions': factor_risk_contributions
        }
    
    def regime_based_attribution(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series,
        regime_indicator: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Regime-based performance attribution.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            regime_indicator: Regime indicator (e.g., 1 for bull, -1 for bear)
            
        Returns:
            Dictionary with regime-based attribution
        """
        # Align dates
        common_dates = strategy_returns.index.intersection(benchmark_returns.index).intersection(regime_indicator.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        regime_aligned = regime_indicator.loc[common_dates]
        
        # Identify regimes
        bull_market = regime_aligned > 0
        bear_market = regime_aligned < 0
        neutral_market = regime_aligned == 0
        
        attribution_by_regime = {}
        
        for regime_name, regime_mask in [('bull', bull_market), ('bear', bear_market), ('neutral', neutral_market)]:
            if regime_mask.any():
                strategy_regime = strategy_aligned[regime_mask]
                benchmark_regime = benchmark_aligned[regime_mask]
                
                # Calculate regime-specific metrics
                excess_return = strategy_regime.mean() - benchmark_regime.mean()
                tracking_error = (strategy_regime - benchmark_regime).std()
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                
                # Beta and alpha
                if len(strategy_regime) > 1:
                    beta, alpha, r_squared, _, _ = stats.linregress(benchmark_regime, strategy_regime)
                else:
                    beta = alpha = r_squared = 0
                
                attribution_by_regime[regime_name] = {
                    'excess_return': excess_return,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': r_squared,
                    'observations': len(strategy_regime)
                }
        
        return attribution_by_regime


def create_benchmark_comparison(
    strategy_returns: pd.Series, 
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Create comprehensive benchmark comparison table.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with comparison metrics
    """
    # Align dates
    common_dates = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_aligned = strategy_returns.loc[common_dates]
    benchmark_aligned = benchmark_returns.loc[common_dates]
    
    # Calculate metrics
    strategy_metrics = {
        'Total Return': (1 + strategy_aligned).prod() - 1,
        'Annualized Return': (1 + strategy_aligned).prod() ** (252 / len(strategy_aligned)) - 1,
        'Volatility': strategy_aligned.std() * np.sqrt(252),
        'Sharpe Ratio': (strategy_aligned.mean() - risk_free_rate/252) / strategy_aligned.std() * np.sqrt(252),
        'Max Drawdown': ((1 + strategy_aligned).cumprod() / (1 + strategy_aligned).cumprod().expanding().max() - 1).min(),
        'Skewness': strategy_aligned.skew(),
        'Kurtosis': strategy_aligned.kurtosis()
    }
    
    benchmark_metrics = {
        'Total Return': (1 + benchmark_aligned).prod() - 1,
        'Annualized Return': (1 + benchmark_aligned).prod() ** (252 / len(benchmark_aligned)) - 1,
        'Volatility': benchmark_aligned.std() * np.sqrt(252),
        'Sharpe Ratio': (benchmark_aligned.mean() - risk_free_rate/252) / benchmark_aligned.std() * np.sqrt(252),
        'Max Drawdown': ((1 + benchmark_aligned).cumprod() / (1 + benchmark_aligned).cumprod().expanding().max() - 1).min(),
        'Skewness': benchmark_aligned.skew(),
        'Kurtosis': benchmark_aligned.kurtosis()
    }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Strategy': strategy_metrics,
        'Benchmark': benchmark_metrics
    })
    
    # Add difference column
    comparison_df['Difference'] = comparison_df['Strategy'] - comparison_df['Benchmark']
    
    return comparison_df
