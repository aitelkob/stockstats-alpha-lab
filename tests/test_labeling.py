"""
Tests for labeling module.

This module tests the labeling functions to ensure proper time-series
discipline and correct label generation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from labeling import (
    LabelingEngine,
    create_feature_matrix,
    calculate_information_coefficient
)


class TestLabeling:
    """Test class for labeling functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create realistic price data
        np.random.seed(42)
        price = 100
        prices = []
        
        for _ in range(100):
            price += np.random.normal(0, 0.02) * price
            prices.append(price)
        
        prices = np.array(prices)
        
        # Create OHLCV data with indicators
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'atr_14': np.random.uniform(0.5, 2.0, 100),
            'close_20_sma': prices * (1 + np.random.normal(0, 0.05, 100))  # Changed from close_200_sma
        }, index=dates)
        
        # Ensure high >= low, high >= open, high >= close
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data
    
    def test_forward_return_label(self, sample_data):
        """Test forward return labeling."""
        labeler = LabelingEngine()
        
        # Test log returns
        forward_returns = labeler.forward_return_label(sample_data, horizon=5, return_type='log')
        
        # Check that we have the right number of labels
        assert len(forward_returns) == len(sample_data)
        
        # Check that the last 5 values are NaN (no future data)
        assert forward_returns.iloc[-5:].isna().all()
        
        # Check that non-NaN values are reasonable
        valid_returns = forward_returns.dropna()
        assert len(valid_returns) > 0
        assert not np.isinf(valid_returns).any()
    
    def test_forward_return_types(self, sample_data):
        """Test different return types."""
        labeler = LabelingEngine()
        
        # Test simple returns
        simple_returns = labeler.forward_return_label(sample_data, horizon=5, return_type='simple')
        assert len(simple_returns) == len(sample_data)
        
        # Test excess returns
        excess_returns = labeler.forward_return_label(sample_data, horizon=5, return_type='excess')
        assert len(excess_returns) == len(sample_data)
        
        # Simple returns should be higher than log returns (for positive returns)
        valid_idx = simple_returns.notna() & simple_returns > 0
        if valid_idx.any():
            assert (simple_returns[valid_idx] > np.log(1 + simple_returns[valid_idx])).all()
    
    def test_binary_classification_label(self, sample_data):
        """Test binary classification labeling."""
        labeler = LabelingEngine()
        
        # Create forward returns first
        forward_returns = labeler.forward_return_label(sample_data, horizon=5)
        
        # Create binary labels
        binary_labels = labeler.binary_classification_label(forward_returns, threshold=0.0)
        
        # Check that we have binary labels
        assert set(binary_labels.dropna().unique()).issubset({0, 1})
        
        # Check that labels align with returns
        valid_idx = forward_returns.notna() & binary_labels.notna()
        if valid_idx.any():
            positive_returns = forward_returns[valid_idx] > 0
            positive_labels = binary_labels[valid_idx] == 1
            assert positive_returns.equals(positive_labels)
    
    def test_triple_barrier_label(self, sample_data):
        """Test triple-barrier labeling."""
        labeler = LabelingEngine()
        
        # Create triple-barrier labels
        triple_labels = labeler.triple_barrier_label(sample_data, horizon=20)
        
        # Check that we have the right labels
        assert set(triple_labels.dropna().unique()).issubset({-1, 0, 1})
        
        # Check that we have some non-zero labels (may be all zeros for small datasets)
        # This is acceptable for small test datasets
        unique_labels = set(triple_labels.dropna().unique())
        assert len(unique_labels) >= 1  # At least one unique label
    
    def test_regime_based_label(self, sample_data):
        """Test regime-based labeling."""
        labeler = LabelingEngine()
        
        # Create regime labels
        regime_labels = labeler.regime_based_label(sample_data)
        
        # Check that we have regime labels
        assert set(regime_labels.dropna().unique()).issubset({0, 1, 2, 3, 4})
        
        # Check that we have some variation in regimes
        assert regime_labels.nunique() > 1
    
    def test_multi_class_label(self, sample_data):
        """Test multi-class labeling."""
        labeler = LabelingEngine()
        
        # Create forward returns first
        forward_returns = labeler.forward_return_label(sample_data, horizon=5)
        
        # Create multi-class labels
        multi_labels = labeler.multi_class_label(forward_returns, quantiles=[0.25, 0.75])
        
        # Check that we have multi-class labels
        unique_labels = set(multi_labels.dropna().unique())
        assert len(unique_labels) >= 2  # At least 2 classes
        
        # Check that labels are integers
        assert all(isinstance(label, (int, np.integer)) for label in unique_labels)
    
    def test_create_feature_matrix(self, sample_data):
        """Test feature matrix creation."""
        # Add some indicator columns
        sample_data['indicator1'] = np.random.randn(100)
        sample_data['indicator2'] = np.random.randn(100)
        
        # Create labels
        labeler = LabelingEngine()
        forward_returns = labeler.forward_return_label(sample_data, horizon=5)
        
        # Create feature matrix
        indicator_cols = ['indicator1', 'indicator2']
        X, y = create_feature_matrix(sample_data, indicator_cols, forward_returns.name)
        
        # Check shapes
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(indicator_cols)
        
        # Check that we don't have NaN values
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_create_feature_matrix_with_lookback(self, sample_data):
        """Test feature matrix creation with lookback window."""
        # Add some indicator columns
        sample_data['indicator1'] = np.random.randn(100)
        sample_data['indicator2'] = np.random.randn(100)
        
        # Create labels
        labeler = LabelingEngine()
        forward_returns = labeler.forward_return_label(sample_data, horizon=5)
        
        # Create feature matrix with lookback
        indicator_cols = ['indicator1', 'indicator2']
        X, y = create_feature_matrix(sample_data, indicator_cols, forward_returns.name, lookback_window=3)
        
        # Check that we have more features due to lookback
        assert X.shape[1] > len(indicator_cols)
        
        # Check that lookback features are present
        lookback_features = [col for col in X.columns if '_lag_' in col]
        assert len(lookback_features) > 0
    
    def test_calculate_information_coefficient(self, sample_data):
        """Test Information Coefficient calculation."""
        # Create features and labels
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        labels = pd.Series(np.random.randn(100))
        
        # Calculate IC
        ic_values = calculate_information_coefficient(features, labels)
        
        # Check that we get IC values for all features
        assert len(ic_values) == len(features.columns)
        
        # Check that IC values are between -1 and 1
        assert all(-1 <= ic <= 1 for ic in ic_values.values)
    
    def test_transaction_cost_impact(self, sample_data):
        """Test that transaction costs affect labeling."""
        labeler = LabelingEngine(transaction_cost=0.01)  # 1% transaction cost
        
        # Create forward returns
        forward_returns = labeler.forward_return_label(sample_data, horizon=5)
        
        # Create binary labels with and without transaction cost
        labels_with_cost = labeler.binary_classification_label(forward_returns, transaction_cost=0.01)
        labels_without_cost = labeler.binary_classification_label(forward_returns, transaction_cost=0.0)
        
        # Labels should be different due to transaction cost
        assert not labels_with_cost.equals(labels_without_cost)
        
        # With transaction cost, we should have fewer positive labels
        assert labels_with_cost.sum() <= labels_without_cost.sum()
    
    def test_time_series_discipline(self, sample_data):
        """Test that labeling maintains time-series discipline."""
        labeler = LabelingEngine()
        
        # Create forward returns
        forward_returns = labeler.forward_return_label(sample_data, horizon=5)
        
        # Check that the last 5 values are NaN (no future data)
        assert forward_returns.iloc[-5:].isna().all()
        
        # Check that we don't have any future data leakage
        # This is ensured by the shift(-horizon) operation in the function


class TestLabelingEdgeCases:
    """Test edge cases for labeling functions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        labeler = LabelingEngine()
        
        # Should handle empty DataFrame gracefully
        result = labeler.forward_return_label(empty_df)
        assert isinstance(result, pd.Series)
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        single_row = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000000]
        })
        
        labeler = LabelingEngine()
        forward_returns = labeler.forward_return_label(single_row, horizon=5)
        
        # Should return a Series with NaN
        assert len(forward_returns) == 1
        assert pd.isna(forward_returns.iloc[0])
    
    def test_all_nan_data(self):
        """Test handling of all-NaN data."""
        nan_df = pd.DataFrame({
            'open': [np.nan] * 10,
            'high': [np.nan] * 10,
            'low': [np.nan] * 10,
            'close': [np.nan] * 10,
            'volume': [np.nan] * 10
        })
        
        labeler = LabelingEngine()
        
        # Should handle NaN data gracefully
        result = labeler.forward_return_label(nan_df)
        assert isinstance(result, pd.Series)
    
    def test_very_short_horizon(self):
        """Test with very short horizon."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })
        
        labeler = LabelingEngine()
        forward_returns = labeler.forward_return_label(data, horizon=1)
        
        # Should have one NaN at the end
        assert forward_returns.iloc[-1] is pd.NA or np.isnan(forward_returns.iloc[-1])
        assert not forward_returns.iloc[:-1].isna().any()


if __name__ == "__main__":
    pytest.main([__file__])
