"""
Advanced Risk Metrics Module

This module provides comprehensive risk analysis tools including VaR, CVaR,
drawdown analysis, and portfolio risk metrics for quantitative finance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Advanced risk analysis and metrics calculation."""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize risk analyzer.
        
        Args:
            confidence_levels: List of confidence levels for VaR/CVaR calculations
        """
        self.confidence_levels = confidence_levels
        self.risk_metrics = {}
    
    def calculate_var(self, returns: pd.Series, method: str = "historical") -> Dict[float, float]:
        """
        Calculate Value at Risk (VaR) using different methods.
        
        Args:
            returns: Series of returns
            method: Method to use ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary mapping confidence levels to VaR values
        """
        var_results = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            
            if method == "historical":
                var_value = np.percentile(returns.dropna(), alpha * 100)
            
            elif method == "parametric":
                mean_return = returns.mean()
                std_return = returns.std()
                var_value = mean_return + std_return * stats.norm.ppf(alpha)
            
            elif method == "monte_carlo":
                # Monte Carlo simulation
                n_simulations = 10000
                mean_return = returns.mean()
                std_return = returns.std()
                
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var_value = np.percentile(simulated_returns, alpha * 100)
            
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            var_results[conf_level] = var_value
        
        return var_results
    
    def calculate_cvar(self, returns: pd.Series, method: str = "historical") -> Dict[float, float]:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
            method: Method to use ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary mapping confidence levels to CVaR values
        """
        cvar_results = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            
            if method == "historical":
                var_value = np.percentile(returns.dropna(), alpha * 100)
                cvar_value = returns[returns <= var_value].mean()
            
            elif method == "parametric":
                mean_return = returns.mean()
                std_return = returns.std()
                var_value = mean_return + std_return * stats.norm.ppf(alpha)
                
                # CVaR for normal distribution
                phi_alpha = stats.norm.pdf(stats.norm.ppf(alpha))
                cvar_value = mean_return - std_return * (phi_alpha / alpha)
            
            elif method == "monte_carlo":
                n_simulations = 10000
                mean_return = returns.mean()
                std_return = returns.std()
                
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var_value = np.percentile(simulated_returns, alpha * 100)
                cvar_value = simulated_returns[simulated_returns <= var_value].mean()
            
            else:
                raise ValueError(f"Unknown CVaR method: {method}")
            
            cvar_results[conf_level] = cvar_value
        
        return cvar_results
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            prices: Series of prices
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Find the start and end of maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_start = prices[:max_dd_idx].idxmax()
        max_dd_end = max_dd_idx
        
        # Duration of maximum drawdown
        max_dd_duration = (max_dd_end - max_dd_start).days if hasattr(max_dd_end - max_dd_start, 'days') else len(prices[max_dd_start:max_dd_end])
        
        # Recovery time (time to reach new high after max drawdown)
        recovery_time = None
        if max_dd_end < len(prices) - 1:
            post_dd_prices = prices[max_dd_end:]
            post_dd_max = post_dd_prices.expanding().max()
            recovery_idx = post_dd_prices[post_dd_prices >= prices[max_dd_start]].index
            if len(recovery_idx) > 0:
                recovery_time = (recovery_idx[0] - max_dd_end).days if hasattr(recovery_idx[0] - max_dd_end, 'days') else len(prices[max_dd_end:recovery_idx[0]])
        
        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean()
        
        # Drawdown frequency
        dd_frequency = (drawdown < 0).sum() / len(drawdown)
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_start': max_dd_start,
            'max_drawdown_end': max_dd_end,
            'max_drawdown_duration': max_dd_duration,
            'recovery_time': recovery_time,
            'avg_drawdown': avg_dd,
            'drawdown_frequency': dd_frequency
        }
    
    def calculate_rolling_var(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling VaR.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Series of rolling VaR values
        """
        rolling_var = returns.rolling(window=window).apply(
            lambda x: np.percentile(x.dropna(), 5), raw=False
        )
        return rolling_var
    
    def calculate_rolling_cvar(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling CVaR.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Series of rolling CVaR values
        """
        def calculate_cvar_window(x):
            if len(x.dropna()) < 10:  # Need minimum data points
                return np.nan
            var_95 = np.percentile(x.dropna(), 5)
            return x[x <= var_95].mean()
        
        rolling_cvar = returns.rolling(window=window).apply(calculate_cvar_window, raw=False)
        return rolling_cvar
    
    def calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate various volatility metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with volatility metrics
        """
        # Basic volatility metrics
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.std()
        
        # GARCH-like volatility clustering
        squared_returns = returns ** 2
        vol_clustering = squared_returns.autocorr(lag=1)
        
        # Asymmetric volatility (leverage effect)
        negative_returns = returns[returns < 0]
        positive_returns = returns[returns > 0]
        
        neg_vol = negative_returns.std() if len(negative_returns) > 0 else 0
        pos_vol = positive_returns.std() if len(positive_returns) > 0 else 0
        leverage_effect = neg_vol - pos_vol
        
        # Volatility percentiles
        vol_95 = rolling_vol.quantile(0.95)
        vol_5 = rolling_vol.quantile(0.05)
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'volatility_of_volatility': vol_of_vol,
            'volatility_clustering': vol_clustering,
            'leverage_effect': leverage_effect,
            'volatility_95th_percentile': vol_95,
            'volatility_5th_percentile': vol_5,
            'volatility_skewness': rolling_vol.skew(),
            'volatility_kurtosis': rolling_vol.kurtosis()
        }
    
    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tail risk and extreme value metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with tail risk metrics
        """
        # Tail ratio (ratio of 95th to 5th percentile)
        tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05))
        
        # Expected shortfall beyond VaR
        var_95 = np.percentile(returns.dropna(), 5)
        var_99 = np.percentile(returns.dropna(), 1)
        
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Tail expectation ratio
        tail_expectation_ratio = es_99 / es_95 if es_95 != 0 else np.nan
        
        # Extreme value index (Hill estimator)
        sorted_returns = returns.dropna().sort_values()
        n = len(sorted_returns)
        k = max(1, int(0.1 * n))  # Use top 10% of observations
        
        if k > 1:
            log_ratios = np.log(sorted_returns.iloc[-k:]) - np.log(sorted_returns.iloc[-(k+1)])
            hill_estimator = 1 / log_ratios.mean()
        else:
            hill_estimator = np.nan
        
        # Maximum likelihood estimation of tail index
        try:
            # Fit generalized Pareto distribution to exceedances
            threshold = returns.quantile(0.95)
            exceedances = returns[returns > threshold] - threshold
            
            if len(exceedances) > 10:
                # MLE for GPD parameters
                def neg_log_likelihood(params):
                    xi, sigma = params
                    if sigma <= 0:
                        return np.inf
                    if xi == 0:
                        return len(exceedances) * np.log(sigma) + exceedances.sum() / sigma
                    else:
                        if xi < 0 and exceedances.max() > -sigma / xi:
                            return np.inf
                        return len(exceedances) * np.log(sigma) + (1 + 1/xi) * np.log(1 + xi * exceedances / sigma).sum()
                
                result = minimize(neg_log_likelihood, [0.1, 0.1], method='L-BFGS-B')
                if result.success:
                    tail_index = result.x[0]
                else:
                    tail_index = np.nan
            else:
                tail_index = np.nan
        except:
            tail_index = np.nan
        
        return {
            'tail_ratio': tail_ratio,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'tail_expectation_ratio': tail_expectation_ratio,
            'hill_estimator': hill_estimator,
            'tail_index_mle': tail_index,
            'extreme_value_95': returns.quantile(0.95),
            'extreme_value_99': returns.quantile(0.99),
            'extreme_value_99.9': returns.quantile(0.999)
        }
    
    def calculate_liquidity_metrics(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """
        Calculate liquidity risk metrics.
        
        Args:
            prices: Series of prices
            volumes: Series of volumes
            
        Returns:
            Dictionary with liquidity metrics
        """
        # Price impact (Amihud illiquidity measure)
        returns = prices.pct_change().dropna()
        volumes_adj = volumes.iloc[1:]  # Align with returns
        
        amihud_illiquidity = abs(returns) / volumes_adj
        amihud_illiquidity = amihud_illiquidity.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Roll's measure of effective bid-ask spread
        price_changes = prices.diff().dropna()
        roll_measure = 2 * np.sqrt(-price_changes.autocorr(lag=1)) if price_changes.autocorr(lag=1) < 0 else 0
        
        # Volume-weighted average price (VWAP) deviation
        vwap = (prices * volumes).sum() / volumes.sum()
        vwap_deviation = abs(prices - vwap).mean()
        
        # Volume volatility
        volume_volatility = volumes.pct_change().std()
        
        # Price-volume correlation
        price_volume_corr = prices.corr(volumes)
        
        # Liquidity ratio (volume / price range)
        price_range = (prices.rolling(5).max() - prices.rolling(5).min()) / prices.rolling(5).mean()
        liquidity_ratio = (volumes / price_range).mean()
        
        return {
            'amihud_illiquidity': amihud_illiquidity.mean(),
            'roll_measure': roll_measure,
            'vwap_deviation': vwap_deviation,
            'volume_volatility': volume_volatility,
            'price_volume_correlation': price_volume_corr,
            'liquidity_ratio': liquidity_ratio,
            'avg_daily_volume': volumes.mean(),
            'volume_trend': volumes.pct_change().mean()
        }
    
    def calculate_regime_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate regime-based risk metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with regime risk metrics
        """
        # Regime identification using rolling statistics
        window = 60  # 3-month window
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        # Z-score for regime identification
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Define regimes
        high_vol_regime = z_scores.abs() > 2
        low_vol_regime = z_scores.abs() < 0.5
        
        # Regime persistence
        regime_changes = high_vol_regime.diff().abs().sum()
        regime_persistence = 1 - (regime_changes / len(returns))
        
        # Regime-specific metrics
        high_vol_returns = returns[high_vol_regime]
        low_vol_returns = returns[low_vol_regime]
        
        high_vol_metrics = {
            'high_vol_frequency': high_vol_regime.mean(),
            'high_vol_mean_return': high_vol_returns.mean() if len(high_vol_returns) > 0 else 0,
            'high_vol_volatility': high_vol_returns.std() if len(high_vol_returns) > 0 else 0,
            'high_vol_skewness': high_vol_returns.skew() if len(high_vol_returns) > 0 else 0
        }
        
        low_vol_metrics = {
            'low_vol_frequency': low_vol_regime.mean(),
            'low_vol_mean_return': low_vol_returns.mean() if len(low_vol_returns) > 0 else 0,
            'low_vol_volatility': low_vol_returns.std() if len(low_vol_returns) > 0 else 0,
            'low_vol_skewness': low_vol_returns.skew() if len(low_vol_returns) > 0 else 0
        }
        
        # Regime transition probabilities
        regime_sequence = high_vol_regime.astype(int)
        transitions = np.zeros((2, 2))
        
        for i in range(1, len(regime_sequence)):
            current = regime_sequence.iloc[i]
            previous = regime_sequence.iloc[i-1]
            transitions[previous, current] += 1
        
        # Normalize transition matrix
        row_sums = transitions.sum(axis=1)
        transition_matrix = transitions / row_sums[:, np.newaxis]
        
        return {
            'regime_persistence': regime_persistence,
            'regime_transition_matrix': transition_matrix.tolist(),
            **high_vol_metrics,
            **low_vol_metrics
        }
    
    def calculate_comprehensive_risk_metrics(self, returns: pd.Series, prices: pd.Series = None, volumes: pd.Series = None) -> Dict[str, Union[float, Dict]]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of returns
            prices: Series of prices (optional)
            volumes: Series of volumes (optional)
            
        Returns:
            Dictionary with all risk metrics
        """
        logger.info("Calculating comprehensive risk metrics...")
        
        risk_metrics = {}
        
        # Basic risk metrics
        risk_metrics['var'] = self.calculate_var(returns)
        risk_metrics['cvar'] = self.calculate_cvar(returns)
        risk_metrics['max_drawdown'] = self.calculate_maximum_drawdown(prices if prices is not None else (1 + returns).cumprod())
        risk_metrics['volatility'] = self.calculate_volatility_metrics(returns)
        risk_metrics['tail_risk'] = self.calculate_tail_risk_metrics(returns)
        risk_metrics['regime_risk'] = self.calculate_regime_risk_metrics(returns)
        
        # Liquidity metrics (if data available)
        if prices is not None and volumes is not None:
            risk_metrics['liquidity'] = self.calculate_liquidity_metrics(prices, volumes)
        
        # Rolling metrics
        risk_metrics['rolling_var'] = self.calculate_rolling_var(returns)
        risk_metrics['rolling_cvar'] = self.calculate_rolling_cvar(returns)
        
        # Store in instance
        self.risk_metrics = risk_metrics
        
        logger.info("Risk metrics calculation completed")
        return risk_metrics


def calculate_portfolio_var(weights: np.ndarray, returns: pd.DataFrame, confidence_level: float = 0.95) -> float:
    """
    Calculate portfolio VaR using variance-covariance method.
    
    Args:
        weights: Portfolio weights
        returns: DataFrame of asset returns
        confidence_level: Confidence level for VaR
        
    Returns:
        Portfolio VaR
    """
    portfolio_return = (weights * returns).sum(axis=1)
    portfolio_std = np.sqrt(weights.T @ returns.cov() @ weights)
    
    alpha = 1 - confidence_level
    var = portfolio_return.mean() + portfolio_std * stats.norm.ppf(alpha)
    
    return var


def calculate_portfolio_cvar(weights: np.ndarray, returns: pd.DataFrame, confidence_level: float = 0.95) -> float:
    """
    Calculate portfolio CVaR.
    
    Args:
        weights: Portfolio weights
        returns: DataFrame of asset returns
        confidence_level: Confidence level for CVaR
        
    Returns:
        Portfolio CVaR
    """
    portfolio_return = (weights * returns).sum(axis=1)
    var = calculate_portfolio_var(weights, returns, confidence_level)
    cvar = portfolio_return[portfolio_return <= var].mean()
    
    return cvar


def optimize_portfolio_risk(returns: pd.DataFrame, target_return: float = None, risk_measure: str = "var") -> Dict:
    """
    Optimize portfolio for risk minimization.
    
    Args:
        returns: DataFrame of asset returns
        target_return: Target portfolio return (optional)
        risk_measure: Risk measure to minimize ('var', 'cvar', 'volatility')
        
    Returns:
        Dictionary with optimization results
    """
    n_assets = len(returns.columns)
    
    # Expected returns
    expected_returns = returns.mean()
    
    # Covariance matrix
    cov_matrix = returns.cov()
    
    # Objective function
    def objective(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if risk_measure == "var":
            # Minimize VaR (approximated by mean - 1.65 * std)
            return -(portfolio_return - 1.65 * portfolio_std)
        elif risk_measure == "cvar":
            # Minimize CVaR (approximated by mean - 2.33 * std)
            return -(portfolio_return - 2.33 * portfolio_std)
        elif risk_measure == "volatility":
            return portfolio_std
        else:
            raise ValueError(f"Unknown risk measure: {risk_measure}")
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    if target_return is not None:
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: np.sum(x * expected_returns) - target_return
        })
    
    # Bounds (long-only)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess
    x0 = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_std = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': portfolio_return / portfolio_std if portfolio_std > 0 else 0,
            'success': True
        }
    else:
        return {
            'weights': None,
            'expected_return': None,
            'volatility': None,
            'sharpe_ratio': None,
            'success': False,
            'message': result.message
        }
