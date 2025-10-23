"""
Tests for Benchmarking and Performance Attribution Module

This module tests the benchmarking functionality including
performance attribution, factor analysis, and style analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
import sys
sys.path.append('src')

from benchmarking import (
    BenchmarkAnalyzer,
    PerformanceAttributor,
    create_benchmark_comparison
)


class TestBenchmarkAnalyzer:
    """Test the BenchmarkAnalyzer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'Strategy': np.random.normal(0.001, 0.02, 100),
            'Benchmark': np.random.normal(0.0008, 0.018, 100),
            'Asset1': np.random.normal(0.0012, 0.025, 100),
            'Asset2': np.random.normal(0.0009, 0.022, 100)
        }, index=dates)
    
    @pytest.fixture
    def benchmark_analyzer(self):
        """Create BenchmarkAnalyzer instance."""
        return BenchmarkAnalyzer()
    
    def test_add_benchmark(self, benchmark_analyzer):
        """Test adding a benchmark."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark_analyzer.add_benchmark('Test Benchmark', returns, 'Test description')
        
        assert 'Test Benchmark' in benchmark_analyzer.benchmark_data
        assert benchmark_analyzer.benchmark_data['Test Benchmark']['description'] == 'Test description'
    
    def test_calculate_relative_performance(self, benchmark_analyzer, sample_returns):
        """Test relative performance calculation."""
        # Add benchmark
        benchmark_analyzer.add_benchmark('Benchmark', sample_returns['Benchmark'])
        
        # Calculate relative performance
        rel_perf = benchmark_analyzer.calculate_relative_performance(
            sample_returns['Strategy'], 'Benchmark'
        )
        
        assert isinstance(rel_perf, dict)
        assert 'benchmark' in rel_perf
        assert 'total_excess_return' in rel_perf
        assert 'annualized_excess_return' in rel_perf
        assert 'tracking_error' in rel_perf
        assert 'information_ratio' in rel_perf
        assert 'beta' in rel_perf
        assert 'alpha' in rel_perf
        assert 'alpha_annualized' in rel_perf
        assert 'r_squared' in rel_perf
        assert 'up_capture' in rel_perf
        assert 'down_capture' in rel_perf
        assert 'win_rate' in rel_perf
        assert 'max_relative_drawdown' in rel_perf
        
        # Check that benchmark name is correct
        assert rel_perf['benchmark'] == 'Benchmark'
    
    def test_calculate_rolling_attribution(self, benchmark_analyzer, sample_returns):
        """Test rolling attribution calculation."""
        # Add benchmark
        benchmark_analyzer.add_benchmark('Benchmark', sample_returns['Benchmark'])
        
        # Calculate rolling attribution
        rolling_attr = benchmark_analyzer.calculate_rolling_attribution(
            sample_returns['Strategy'], 'Benchmark', window=30
        )
        
        assert isinstance(rolling_attr, pd.DataFrame)
        assert 'alpha' in rolling_attr.columns
        assert 'beta' in rolling_attr.columns
        assert 'r_squared' in rolling_attr.columns
        assert 'tracking_error' in rolling_attr.columns
        assert 'information_ratio' in rolling_attr.columns
        
        # Should have fewer rows than original due to rolling window
        assert len(rolling_attr) < len(sample_returns)
    
    def test_factor_attribution(self, benchmark_analyzer, sample_returns):
        """Test factor attribution calculation."""
        # Create factor returns
        factor_returns = pd.DataFrame({
            'Market': sample_returns['Benchmark'],
            'Size': np.random.normal(0, 0.01, 100),
            'Value': np.random.normal(0, 0.01, 100)
        }, index=sample_returns.index)
        
        # Calculate factor attribution
        factor_attr = benchmark_analyzer.factor_attribution(
            sample_returns['Strategy'], factor_returns
        )
        
        assert isinstance(factor_attr, dict)
        assert 'factor_loadings' in factor_attr
        assert 'factor_contributions' in factor_attr
        assert 'alpha' in factor_attr
        assert 'residual_return' in factor_attr
        assert 'r_squared' in factor_attr
        assert 'explained_variance' in factor_attr
        assert 'residual_variance' in factor_attr
        assert 'factor_risk_contributions' in factor_attr
        
        # Check factor loadings
        assert len(factor_attr['factor_loadings']) == len(factor_returns.columns)
    
    def test_style_analysis(self, benchmark_analyzer, sample_returns):
        """Test style analysis calculation."""
        # Create style factors
        style_factors = pd.DataFrame({
            'Large Cap': np.random.normal(0, 0.01, 100),
            'Small Cap': np.random.normal(0, 0.01, 100),
            'Growth': np.random.normal(0, 0.01, 100),
            'Value': np.random.normal(0, 0.01, 100)
        }, index=sample_returns.index)
        
        # Calculate style analysis
        style_analysis = benchmark_analyzer.style_analysis(
            sample_returns['Strategy'], style_factors
        )
        
        assert isinstance(style_analysis, dict)
        assert 'style_weights' in style_analysis
        assert 'r_squared' in style_analysis
        assert 'tracking_error' in style_analysis
        assert 'success' in style_analysis
        
        if style_analysis['success']:
            # Check that style weights sum to 1
            weight_sum = sum(style_analysis['style_weights'].values())
            assert abs(weight_sum - 1.0) < 1e-6
    
    def test_calculate_attribution_metrics(self, benchmark_analyzer, sample_returns):
        """Test comprehensive attribution metrics calculation."""
        # Add benchmark
        benchmark_analyzer.add_benchmark('Benchmark', sample_returns['Benchmark'])
        
        # Calculate attribution metrics
        attribution = benchmark_analyzer.calculate_attribution_metrics(
            sample_returns['Strategy'], 'Benchmark'
        )
        
        assert isinstance(attribution, dict)
        assert 'relative_performance' in attribution
        assert 'rolling_attribution' in attribution
        
        # Check that results are stored
        assert 'Benchmark' in benchmark_analyzer.attribution_results


class TestPerformanceAttributor:
    """Test the PerformanceAttributor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return {
            'portfolio_returns': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
            'benchmark_returns': pd.Series(np.random.normal(0.0008, 0.018, 100), index=dates),
            'sector_returns': pd.DataFrame({
                'Technology': np.random.normal(0.001, 0.02, 100),
                'Healthcare': np.random.normal(0.0008, 0.015, 100),
                'Finance': np.random.normal(0.0009, 0.019, 100)
            }, index=dates),
            'portfolio_weights': pd.Series([0.4, 0.3, 0.3], index=['Technology', 'Healthcare', 'Finance']),
            'benchmark_weights': pd.Series([0.5, 0.25, 0.25], index=['Technology', 'Healthcare', 'Finance']),
            'factor_returns': pd.DataFrame({
                'Market': np.random.normal(0.0008, 0.018, 100),
                'Size': np.random.normal(0, 0.01, 100),
                'Value': np.random.normal(0, 0.01, 100)
            }, index=dates),
            'regime_indicator': pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)
        }
    
    @pytest.fixture
    def attributor(self):
        """Create PerformanceAttributor instance."""
        return PerformanceAttributor()
    
    def test_brinson_attribution(self, attributor, sample_data):
        """Test Brinson attribution analysis."""
        brinson_attr = attributor.brinson_attribution(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns'],
            sample_data['sector_returns'],
            sample_data['portfolio_weights'],
            sample_data['benchmark_weights']
        )
        
        assert isinstance(brinson_attr, dict)
        assert 'allocation_effect' in brinson_attr
        assert 'selection_effect' in brinson_attr
        assert 'interaction_effect' in brinson_attr
        assert 'total_attribution' in brinson_attr
        assert 'allocation_by_sector' in brinson_attr
        assert 'selection_by_sector' in brinson_attr
        assert 'interaction_by_sector' in brinson_attr
        
        # Check that total attribution equals sum of effects
        total = (brinson_attr['allocation_effect'] + 
                brinson_attr['selection_effect'] + 
                brinson_attr['interaction_effect'])
        assert abs(total - brinson_attr['total_attribution']) < 1e-10
    
    def test_factor_attribution_decomposition(self, attributor, sample_data):
        """Test factor attribution decomposition."""
        factor_attr = attributor.factor_attribution_decomposition(
            sample_data['portfolio_returns'],
            sample_data['factor_returns']
        )
        
        assert isinstance(factor_attr, dict)
        assert 'factor_loadings' in factor_attr
        assert 'factor_contributions' in factor_attr
        assert 'alpha' in factor_attr
        assert 'residual_return' in factor_attr
        assert 'r_squared' in factor_attr
        assert 'explained_variance' in factor_attr
        assert 'residual_variance' in factor_attr
        assert 'factor_risk_contributions' in factor_attr
        
        # Check factor loadings
        assert len(factor_attr['factor_loadings']) == len(sample_data['factor_returns'].columns)
    
    def test_factor_attribution_with_loadings(self, attributor, sample_data):
        """Test factor attribution with provided loadings."""
        # Create factor loadings
        factor_loadings = pd.Series([0.8, 0.2, 0.1], index=sample_data['factor_returns'].columns)
        
        factor_attr = attributor.factor_attribution_decomposition(
            sample_data['portfolio_returns'],
            sample_data['factor_returns'],
            factor_loadings=factor_loadings
        )
        
        assert isinstance(factor_attr, dict)
        assert 'factor_loadings' in factor_attr
        assert 'alpha' in factor_attr
    
    def test_regime_based_attribution(self, attributor, sample_data):
        """Test regime-based attribution analysis."""
        regime_attr = attributor.regime_based_attribution(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns'],
            sample_data['regime_indicator']
        )
        
        assert isinstance(regime_attr, dict)
        
        # Check for regime-specific metrics
        for regime in ['bull', 'bear', 'neutral']:
            if regime in regime_attr:
                regime_metrics = regime_attr[regime]
                assert 'excess_return' in regime_metrics
                assert 'tracking_error' in regime_metrics
                assert 'information_ratio' in regime_metrics
                assert 'beta' in regime_metrics
                assert 'alpha' in regime_metrics
                assert 'r_squared' in regime_metrics
                assert 'observations' in regime_metrics


class TestBenchmarkComparison:
    """Test benchmark comparison functions."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return {
            'strategy': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
            'benchmark': pd.Series(np.random.normal(0.0008, 0.018, 100), index=dates)
        }
    
    def test_create_benchmark_comparison(self, sample_returns):
        """Test benchmark comparison table creation."""
        comparison_df = create_benchmark_comparison(
            sample_returns['strategy'],
            sample_returns['benchmark']
        )
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'Strategy' in comparison_df.columns
        assert 'Benchmark' in comparison_df.columns
        assert 'Difference' in comparison_df.columns
        
        # Check that we have the expected metrics
        expected_metrics = ['Total Return', 'Annualized Return', 'Volatility', 
                           'Sharpe Ratio', 'Max Drawdown', 'Skewness', 'Kurtosis']
        
        for metric in expected_metrics:
            assert metric in comparison_df.index
        
        # Check that difference column is calculated correctly
        for metric in expected_metrics:
            strategy_val = comparison_df.loc[metric, 'Strategy']
            benchmark_val = comparison_df.loc[metric, 'Benchmark']
            difference_val = comparison_df.loc[metric, 'Difference']
            assert abs(difference_val - (strategy_val - benchmark_val)) < 1e-10


class TestBenchmarkingEdgeCases:
    """Test edge cases for benchmarking."""
    
    def test_empty_returns(self):
        """Test with empty returns series."""
        analyzer = BenchmarkAnalyzer()
        empty_returns = pd.Series(dtype=float)
        
        # Add empty benchmark
        analyzer.add_benchmark('Empty', empty_returns)
        
        # Should handle gracefully
        rel_perf = analyzer.calculate_relative_performance(empty_returns, 'Empty')
        assert isinstance(rel_perf, dict)
    
    def test_single_return(self):
        """Test with single return value."""
        analyzer = BenchmarkAnalyzer()
        single_return = pd.Series([0.01])
        
        analyzer.add_benchmark('Single', single_return)
        
        rel_perf = analyzer.calculate_relative_performance(single_return, 'Single')
        assert isinstance(rel_perf, dict)
    
    def test_mismatched_dates(self):
        """Test with mismatched date indices."""
        analyzer = BenchmarkAnalyzer()
        
        dates1 = pd.date_range('2023-01-01', periods=50, freq='D')
        dates2 = pd.date_range('2023-02-01', periods=50, freq='D')
        
        returns1 = pd.Series(np.random.normal(0, 0.01, 50), index=dates1)
        returns2 = pd.Series(np.random.normal(0, 0.01, 50), index=dates2)
        
        analyzer.add_benchmark('Mismatched', returns2)
        
        rel_perf = analyzer.calculate_relative_performance(returns1, 'Mismatched')
        assert isinstance(rel_perf, dict)
    
    def test_constant_returns(self):
        """Test with constant returns."""
        analyzer = BenchmarkAnalyzer()
        constant_returns = pd.Series([0.01] * 100)
        
        analyzer.add_benchmark('Constant', constant_returns)
        
        rel_perf = analyzer.calculate_relative_performance(constant_returns, 'Constant')
        assert isinstance(rel_perf, dict)
    
    def test_no_benchmarks(self):
        """Test with no benchmarks added."""
        analyzer = BenchmarkAnalyzer()
        returns = pd.Series([0.01, -0.02, 0.03])
        
        with pytest.raises(ValueError):
            analyzer.calculate_relative_performance(returns)
    
    def test_nonexistent_benchmark(self):
        """Test with nonexistent benchmark name."""
        analyzer = BenchmarkAnalyzer()
        returns = pd.Series([0.01, -0.02, 0.03])
        
        with pytest.raises(ValueError):
            analyzer.calculate_relative_performance(returns, 'Nonexistent')
