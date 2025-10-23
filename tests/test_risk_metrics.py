"""
Tests for Risk Metrics Module

This module tests the comprehensive risk analysis functionality
including VaR, CVaR, drawdown analysis, and portfolio risk metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
import sys
sys.path.append('src')

from risk_metrics import (
    RiskAnalyzer,
    calculate_portfolio_var,
    calculate_portfolio_cvar,
    optimize_portfolio_risk
)


class TestRiskAnalyzer:
    """Test the RiskAnalyzer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample prices data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0.001, 0.02, 252)
        prices = 100 * np.cumprod(1 + returns)
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_volumes(self):
        """Create sample volumes data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        volumes = np.random.randint(1000000, 10000000, 252)
        return pd.Series(volumes, index=dates)
    
    @pytest.fixture
    def risk_analyzer(self):
        """Create RiskAnalyzer instance."""
        return RiskAnalyzer()
    
    def test_calculate_var_historical(self, risk_analyzer, sample_returns):
        """Test VaR calculation using historical method."""
        var_results = risk_analyzer.calculate_var(sample_returns, method="historical")
        
        assert isinstance(var_results, dict)
        assert 0.95 in var_results
        assert 0.99 in var_results
        
        # VaR should be negative (losses)
        assert var_results[0.95] < 0
        assert var_results[0.99] < 0
        
        # 99% VaR should be more extreme than 95% VaR
        assert var_results[0.99] < var_results[0.95]
    
    def test_calculate_var_parametric(self, risk_analyzer, sample_returns):
        """Test VaR calculation using parametric method."""
        var_results = risk_analyzer.calculate_var(sample_returns, method="parametric")
        
        assert isinstance(var_results, dict)
        assert 0.95 in var_results
        assert 0.99 in var_results
        
        # VaR should be negative (losses)
        assert var_results[0.95] < 0
        assert var_results[0.99] < 0
    
    def test_calculate_var_monte_carlo(self, risk_analyzer, sample_returns):
        """Test VaR calculation using Monte Carlo method."""
        var_results = risk_analyzer.calculate_var(sample_returns, method="monte_carlo")
        
        assert isinstance(var_results, dict)
        assert 0.95 in var_results
        assert 0.99 in var_results
        
        # VaR should be negative (losses)
        assert var_results[0.95] < 0
        assert var_results[0.99] < 0
    
    def test_calculate_cvar_historical(self, risk_analyzer, sample_returns):
        """Test CVaR calculation using historical method."""
        cvar_results = risk_analyzer.calculate_cvar(sample_returns, method="historical")
        
        assert isinstance(cvar_results, dict)
        assert 0.95 in cvar_results
        assert 0.99 in cvar_results
        
        # CVaR should be negative (losses)
        assert cvar_results[0.95] < 0
        assert cvar_results[0.99] < 0
        
        # 99% CVaR should be more extreme than 95% CVaR
        assert cvar_results[0.99] < cvar_results[0.95]
    
    def test_calculate_cvar_parametric(self, risk_analyzer, sample_returns):
        """Test CVaR calculation using parametric method."""
        cvar_results = risk_analyzer.calculate_cvar(sample_returns, method="parametric")
        
        assert isinstance(cvar_results, dict)
        assert 0.95 in cvar_results
        assert 0.99 in cvar_results
        
        # CVaR should be negative (losses)
        assert cvar_results[0.95] < 0
        assert cvar_results[0.99] < 0
    
    def test_calculate_maximum_drawdown(self, risk_analyzer, sample_prices):
        """Test maximum drawdown calculation."""
        dd_metrics = risk_analyzer.calculate_maximum_drawdown(sample_prices)
        
        assert isinstance(dd_metrics, dict)
        assert 'max_drawdown' in dd_metrics
        assert 'max_drawdown_pct' in dd_metrics
        assert 'max_drawdown_start' in dd_metrics
        assert 'max_drawdown_end' in dd_metrics
        assert 'max_drawdown_duration' in dd_metrics
        
        # Max drawdown should be negative
        assert dd_metrics['max_drawdown'] <= 0
        assert dd_metrics['max_drawdown_pct'] <= 0
        
        # Duration should be non-negative
        assert dd_metrics['max_drawdown_duration'] >= 0
    
    def test_calculate_rolling_var(self, risk_analyzer, sample_returns):
        """Test rolling VaR calculation."""
        rolling_var = risk_analyzer.calculate_rolling_var(sample_returns, window=30)
        
        assert isinstance(rolling_var, pd.Series)
        assert len(rolling_var) == len(sample_returns)
        
        # Rolling VaR should be mostly negative
        assert rolling_var.dropna().lt(0).sum() > 0
    
    def test_calculate_rolling_cvar(self, risk_analyzer, sample_returns):
        """Test rolling CVaR calculation."""
        rolling_cvar = risk_analyzer.calculate_rolling_cvar(sample_returns, window=30)
        
        assert isinstance(rolling_cvar, pd.Series)
        assert len(rolling_cvar) == len(sample_returns)
    
    def test_calculate_volatility_metrics(self, risk_analyzer, sample_returns):
        """Test volatility metrics calculation."""
        vol_metrics = risk_analyzer.calculate_volatility_metrics(sample_returns)
        
        assert isinstance(vol_metrics, dict)
        assert 'daily_volatility' in vol_metrics
        assert 'annualized_volatility' in vol_metrics
        assert 'volatility_of_volatility' in vol_metrics
        assert 'volatility_clustering' in vol_metrics
        assert 'leverage_effect' in vol_metrics
        
        # Volatility should be positive
        assert vol_metrics['daily_volatility'] > 0
        assert vol_metrics['annualized_volatility'] > 0
    
    def test_calculate_tail_risk_metrics(self, risk_analyzer, sample_returns):
        """Test tail risk metrics calculation."""
        tail_metrics = risk_analyzer.calculate_tail_risk_metrics(sample_returns)
        
        assert isinstance(tail_metrics, dict)
        assert 'tail_ratio' in tail_metrics
        assert 'expected_shortfall_95' in tail_metrics
        assert 'expected_shortfall_99' in tail_metrics
        assert 'tail_expectation_ratio' in tail_metrics
        assert 'hill_estimator' in tail_metrics
        
        # Tail ratio should be positive
        assert tail_metrics['tail_ratio'] > 0
    
    def test_calculate_liquidity_metrics(self, risk_analyzer, sample_prices, sample_volumes):
        """Test liquidity metrics calculation."""
        liq_metrics = risk_analyzer.calculate_liquidity_metrics(sample_prices, sample_volumes)
        
        assert isinstance(liq_metrics, dict)
        assert 'amihud_illiquidity' in liq_metrics
        assert 'roll_measure' in liq_metrics
        assert 'vwap_deviation' in liq_metrics
        assert 'volume_volatility' in liq_metrics
        assert 'price_volume_correlation' in liq_metrics
        assert 'liquidity_ratio' in liq_metrics
    
    def test_calculate_regime_risk_metrics(self, risk_analyzer, sample_returns):
        """Test regime risk metrics calculation."""
        regime_metrics = risk_analyzer.calculate_regime_risk_metrics(sample_returns)
        
        assert isinstance(regime_metrics, dict)
        assert 'regime_persistence' in regime_metrics
        assert 'regime_transition_matrix' in regime_metrics
        assert 'high_vol_frequency' in regime_metrics
        assert 'low_vol_frequency' in regime_metrics
        
        # Persistence should be between 0 and 1
        assert 0 <= regime_metrics['regime_persistence'] <= 1
    
    def test_calculate_comprehensive_risk_metrics(self, risk_analyzer, sample_returns, sample_prices, sample_volumes):
        """Test comprehensive risk metrics calculation."""
        risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(
            sample_returns, sample_prices, sample_volumes
        )
        
        assert isinstance(risk_metrics, dict)
        assert 'var' in risk_metrics
        assert 'cvar' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'volatility' in risk_metrics
        assert 'tail_risk' in risk_metrics
        assert 'regime_risk' in risk_metrics
        assert 'liquidity' in risk_metrics
        assert 'rolling_var' in risk_metrics
        assert 'rolling_cvar' in risk_metrics


class TestPortfolioRiskFunctions:
    """Test portfolio risk calculation functions."""
    
    @pytest.fixture
    def sample_portfolio_returns(self):
        """Create sample portfolio returns data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 100),
            'Asset2': np.random.normal(0.0008, 0.018, 100),
            'Asset3': np.random.normal(0.0012, 0.025, 100)
        }, index=dates)
    
    def test_calculate_portfolio_var(self, sample_portfolio_returns):
        """Test portfolio VaR calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        
        var_95 = calculate_portfolio_var(weights, sample_portfolio_returns, 0.95)
        var_99 = calculate_portfolio_var(weights, sample_portfolio_returns, 0.99)
        
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        
        # VaR should be negative (losses)
        assert var_95 < 0
        assert var_99 < 0
        
        # 99% VaR should be more extreme than 95% VaR
        assert var_99 < var_95
    
    def test_calculate_portfolio_cvar(self, sample_portfolio_returns):
        """Test portfolio CVaR calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        
        cvar_95 = calculate_portfolio_cvar(weights, sample_portfolio_returns, 0.95)
        cvar_99 = calculate_portfolio_cvar(weights, sample_portfolio_returns, 0.99)
        
        assert isinstance(cvar_95, float)
        assert isinstance(cvar_99, float)
        
        # CVaR should be negative (losses)
        assert cvar_95 < 0
        assert cvar_99 < 0
    
    def test_optimize_portfolio_risk(self, sample_portfolio_returns):
        """Test portfolio risk optimization."""
        # Test mean-variance optimization
        result = optimize_portfolio_risk(sample_portfolio_returns, risk_measure="volatility")
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'weights' in result
            assert 'expected_return' in result
            assert 'volatility' in result
            assert 'sharpe_ratio' in result
            
            # Weights should sum to 1
            assert abs(np.sum(result['weights']) - 1.0) < 1e-6
            
            # Weights should be non-negative
            assert np.all(result['weights'] >= 0)
    
    def test_optimize_portfolio_risk_with_target_return(self, sample_portfolio_returns):
        """Test portfolio optimization with target return."""
        target_return = 0.001
        result = optimize_portfolio_risk(
            sample_portfolio_returns, 
            target_return=target_return, 
            risk_measure="volatility"
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # Check that target return is achieved (approximately)
            assert abs(result['expected_return'] - target_return) < 1e-3


class TestRiskMetricsEdgeCases:
    """Test edge cases for risk metrics."""
    
    def test_empty_returns(self):
        """Test with empty returns series."""
        risk_analyzer = RiskAnalyzer()
        empty_returns = pd.Series(dtype=float)
        
        # These should handle empty series gracefully
        var_results = risk_analyzer.calculate_var(empty_returns)
        assert isinstance(var_results, dict)
        
        cvar_results = risk_analyzer.calculate_cvar(empty_returns)
        assert isinstance(cvar_results, dict)
    
    def test_single_return(self):
        """Test with single return value."""
        risk_analyzer = RiskAnalyzer()
        single_return = pd.Series([0.01])
        
        var_results = risk_analyzer.calculate_var(single_return)
        assert isinstance(var_results, dict)
        
        cvar_results = risk_analyzer.calculate_cvar(single_return)
        assert isinstance(cvar_results, dict)
    
    def test_all_zero_returns(self):
        """Test with all zero returns."""
        risk_analyzer = RiskAnalyzer()
        zero_returns = pd.Series([0.0] * 100)
        
        var_results = risk_analyzer.calculate_var(zero_returns)
        assert isinstance(var_results, dict)
        
        cvar_results = risk_analyzer.calculate_cvar(zero_returns)
        assert isinstance(cvar_results, dict)
    
    def test_constant_returns(self):
        """Test with constant returns."""
        risk_analyzer = RiskAnalyzer()
        constant_returns = pd.Series([0.01] * 100)
        
        var_results = risk_analyzer.calculate_var(constant_returns)
        assert isinstance(var_results, dict)
        
        cvar_results = risk_analyzer.calculate_cvar(constant_returns)
        assert isinstance(cvar_results, dict)
    
    def test_invalid_method(self):
        """Test with invalid VaR method."""
        risk_analyzer = RiskAnalyzer()
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        
        with pytest.raises(ValueError):
            risk_analyzer.calculate_var(returns, method="invalid_method")
    
    def test_invalid_confidence_levels(self):
        """Test with invalid confidence levels."""
        risk_analyzer = RiskAnalyzer(confidence_levels=[0.5, 1.5])  # Invalid levels
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        
        # Should handle gracefully or raise appropriate error
        try:
            var_results = risk_analyzer.calculate_var(returns)
            assert isinstance(var_results, dict)
        except (ValueError, AssertionError):
            # Expected for invalid confidence levels
            pass
