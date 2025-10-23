"""
Tests for Advanced Technical Indicators Module

This module tests the advanced technical indicators functionality
including all 8 categories of indicators and custom formulas.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
import sys
sys.path.append('src')

from advanced_indicators import (
    AdvancedIndicatorEngine,
    add_advanced_indicators,
    create_custom_indicator,
    calculate_indicator_correlation,
    find_highly_correlated_indicators
)


class TestAdvancedIndicatorEngine:
    """Test the AdvancedIndicatorEngine class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    @pytest.fixture
    def engine(self):
        """Create AdvancedIndicatorEngine instance."""
        return AdvancedIndicatorEngine()
    
    def test_momentum_indicators(self, engine, sample_data):
        """Test momentum indicators calculation."""
        result = engine.add_momentum_indicators(sample_data)
        
        # Check that result is DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that original columns are preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
        
        # Check for momentum indicators
        momentum_indicators = ['roc_5', 'roc_10', 'roc_20', 'mom_10', 'mom_20', 
                              'wr_14', 'wr_21', 'kdjk', 'kdjd', 'kdjj', 
                              'cci_14', 'cci_20', 'mfi_14']
        
        for indicator in momentum_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_volatility_indicators(self, engine, sample_data):
        """Test volatility indicators calculation."""
        result = engine.add_volatility_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for volatility indicators
        volatility_indicators = ['atr_14', 'atr_21', 'boll', 'boll_ub', 'boll_lb',
                                'boll_wb', 'boll_pb', 'kc_20', 'kc_ub', 'kc_lb',
                                'dc_20', 'dc_ub', 'dc_lb', 'hv_10', 'hv_20', 'hv_30']
        
        for indicator in volatility_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_trend_indicators(self, engine, sample_data):
        """Test trend indicators calculation."""
        result = engine.add_trend_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for trend indicators
        trend_indicators = ['close_5_sma', 'close_10_sma', 'close_20_sma', 'close_50_sma',
                           'close_100_sma', 'close_200_sma', 'close_5_ema', 'close_10_ema',
                           'close_20_ema', 'close_50_ema', 'macd', 'macds', 'macdh',
                           'psar', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base',
                           'ichimoku_con', 'adx_14', 'di_plus', 'di_minus']
        
        for indicator in trend_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_volume_indicators(self, engine, sample_data):
        """Test volume indicators calculation."""
        result = engine.add_volume_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for volume indicators
        volume_indicators = ['obv', 'vroc_5', 'vroc_10', 'vma_5', 'vma_10', 'vma_20',
                            'ad', 'cmf_20', 'vpt']
        
        for indicator in volume_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_custom_indicators(self, engine, sample_data):
        """Test custom indicators calculation."""
        result = engine.add_custom_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for custom indicators
        custom_indicators = ['price_position', 'volatility_ratio', 'price_acceleration',
                            'volume_price_divergence', 'trend_strength', 'resistance_level',
                            'support_level', 'price_vs_resistance', 'price_vs_support',
                            'market_regime', 'volatility_regime']
        
        for indicator in custom_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_fractal_indicators(self, engine, sample_data):
        """Test fractal indicators calculation."""
        result = engine.add_fractal_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for fractal indicators
        fractal_indicators = ['fractal_high', 'fractal_low', 'pivot_point', 'resistance_1',
                             'support_1', 'resistance_2', 'support_2', 'fib_23.6', 'fib_38.2',
                             'fib_50.0', 'fib_61.8', 'fib_78.6']
        
        for indicator in fractal_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_cyclical_indicators(self, engine, sample_data):
        """Test cyclical indicators calculation."""
        result = engine.add_cyclical_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for cyclical indicators
        cyclical_indicators = ['dpo', 'cycle_period', 'month', 'quarter', 'day_of_week',
                              'day_of_month', 'monthly_seasonality']
        
        for indicator in cyclical_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_microstructure_indicators(self, engine, sample_data):
        """Test microstructure indicators calculation."""
        result = engine.add_market_microstructure_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for microstructure indicators
        microstructure_indicators = ['spread_proxy', 'price_impact', 'order_flow_imbalance',
                                    'vpt', 'ease_of_movement', 'force_index', 'force_index_ema']
        
        for indicator in microstructure_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()
    
    def test_add_all_advanced_indicators(self, engine, sample_data):
        """Test adding all advanced indicators."""
        result = engine.add_all_advanced_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(sample_data.columns)
        
        # Check that original columns are preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
    
    def test_get_indicator_categories(self, engine):
        """Test getting indicator categories."""
        categories = engine.get_indicator_categories()
        
        assert isinstance(categories, dict)
        assert len(categories) > 0
        
        expected_categories = ['momentum', 'volatility', 'trend', 'volume', 'custom',
                              'fractal', 'cyclical', 'microstructure']
        
        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], list)
            assert len(categories[category]) > 0


class TestAdvancedIndicatorFunctions:
    """Test advanced indicator utility functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_add_advanced_indicators(self, sample_data):
        """Test add_advanced_indicators function."""
        result = add_advanced_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(sample_data.columns)
        
        # Check that original columns are preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
    
    def test_create_custom_indicator(self, sample_data):
        """Test create_custom_indicator function."""
        # Test simple custom indicator
        formula = 'close - close.shift(1)'
        name = 'price_change'
        
        result = create_custom_indicator(sample_data, formula, name)
        
        assert isinstance(result, pd.Series)
        assert result.name == name
        assert len(result) == len(sample_data)
    
    def test_calculate_indicator_correlation(self, sample_data):
        """Test calculate_indicator_correlation function."""
        # Add some indicators first
        df_with_indicators = add_advanced_indicators(sample_data)
        
        # Get indicator columns
        indicator_cols = [col for col in df_with_indicators.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Test with subset of indicators
        test_indicators = indicator_cols[:10] if len(indicator_cols) >= 10 else indicator_cols
        
        if test_indicators:
            corr_matrix = calculate_indicator_correlation(df_with_indicators, test_indicators)
            
            assert isinstance(corr_matrix, pd.DataFrame)
            assert corr_matrix.shape[0] == len(test_indicators)
            assert corr_matrix.shape[1] == len(test_indicators)
    
    def test_find_highly_correlated_indicators(self, sample_data):
        """Test find_highly_correlated_indicators function."""
        # Add some indicators first
        df_with_indicators = add_advanced_indicators(sample_data)
        
        # Test with different thresholds
        for threshold in [0.8, 0.9, 0.95]:
            correlated = find_highly_correlated_indicators(df_with_indicators, threshold)
            
            assert isinstance(correlated, list)
            
            # Check that all correlations are above threshold
            for indicator1, indicator2, corr in correlated:
                assert abs(corr) >= threshold
                assert isinstance(indicator1, str)
                assert isinstance(indicator2, str)
                assert isinstance(corr, (int, float))


class TestAdvancedIndicatorEdgeCases:
    """Test edge cases for advanced indicators."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        engine = AdvancedIndicatorEngine()
        empty_df = pd.DataFrame()
        
        result = engine.add_all_advanced_indicators(empty_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        engine = AdvancedIndicatorEngine()
        single_row = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
        })
        
        result = engine.add_all_advanced_indicators(single_row)
        assert isinstance(result, pd.DataFrame)
    
    def test_all_nan_data(self):
        """Test with all NaN data."""
        engine = AdvancedIndicatorEngine()
        nan_df = pd.DataFrame({
            'open': [np.nan] * 10,
            'high': [np.nan] * 10,
            'low': [np.nan] * 10,
            'close': [np.nan] * 10,
            'volume': [np.nan] * 10
        })
        
        result = engine.add_all_advanced_indicators(nan_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_missing_columns(self):
        """Test with missing required columns."""
        engine = AdvancedIndicatorEngine()
        incomplete_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        result = engine.add_all_advanced_indicators(incomplete_df)
        assert isinstance(result, pd.DataFrame)
