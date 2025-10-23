"""
Tests for indicators module.

This module tests the technical indicator calculations and validates
them against reference implementations to ensure accuracy.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from indicators import (
    IndicatorEngine,
    add_basic_indicators,
    add_comprehensive_indicators,
    validate_indicators_against_reference,
)


class TestIndicators:
    """Test class for indicator calculations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create realistic price data
        np.random.seed(42)
        price = 100
        prices = []

        for _ in range(100):
            price += np.random.normal(0, 0.02) * price
            prices.append(price)

        prices = np.array(prices)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, 100)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # Ensure high >= low, high >= open, high >= close
        data["high"] = np.maximum(data["high"], data[["open", "close"]].max(axis=1))
        data["low"] = np.minimum(data["low"], data[["open", "close"]].min(axis=1))

        return data

    def test_add_basic_indicators(self, sample_data):
        """Test basic indicator addition."""
        df_with_indicators = add_basic_indicators(sample_data)

        # Check that indicators were added (excluding mstd_20 which we removed)
        expected_indicators = [
            "close_10_sma",
            "close_20_ema",
            "macd",
            "rsi_14",
            "close_10_roc",
            "boll",
            "kdjk",
            "kdjd",
            "kdjj",
            "atr_14",
            "cr",
            "wr_14",
            "log-ret",
        ]

        for indicator in expected_indicators:
            assert (
                indicator in df_with_indicators.columns
            ), f"Indicator {indicator} not found"

        # Check that data shape is preserved
        assert df_with_indicators.shape[0] == sample_data.shape[0]

        # Check that original OHLCV columns are preserved
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df_with_indicators.columns

    def test_add_comprehensive_indicators(self, sample_data):
        """Test comprehensive indicator addition."""
        df_with_indicators = add_comprehensive_indicators(sample_data)

        # Check that many indicators were added
        assert len(df_with_indicators.columns) > len(sample_data.columns) + 20

        # Check that no NaN values in critical indicators
        critical_indicators = ["rsi_14", "macd", "close_20_sma"]
        for indicator in critical_indicators:
            if indicator in df_with_indicators.columns:
                # Allow some NaN values at the beginning due to rolling calculations
                assert df_with_indicators[indicator].notna().sum() > 50

    def test_indicator_engine(self, sample_data):
        """Test IndicatorEngine class."""
        engine = IndicatorEngine()

        # Test with trend indicators only
        df_trend = engine.add_indicators(sample_data, indicator_groups=["trend"])

        # Check that trend indicators were added
        trend_indicators = ["close_5_sma", "close_10_sma", "close_20_sma", "macd"]
        for indicator in trend_indicators:
            if indicator in df_trend.columns:
                assert not df_trend[indicator].isna().all()

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation specifically."""
        df_with_indicators = add_basic_indicators(sample_data)

        if "rsi_14" in df_with_indicators.columns:
            rsi = df_with_indicators["rsi_14"].dropna()

            # RSI should be between 0 and 100
            assert rsi.min() >= 0
            assert rsi.max() <= 100

            # RSI should not be all the same value
            assert rsi.nunique() > 1

    def test_macd_calculation(self, sample_data):
        """Test MACD calculation specifically."""
        df_with_indicators = add_basic_indicators(sample_data)

        if "macd" in df_with_indicators.columns:
            macd = df_with_indicators["macd"].dropna()

            # MACD should have variation
            assert macd.nunique() > 1

            # MACD should not be all NaN
            assert not macd.isna().all()

    def test_moving_averages(self, sample_data):
        """Test moving average calculations."""
        df_with_indicators = add_basic_indicators(sample_data)

        # Test SMA
        if "close_10_sma" in df_with_indicators.columns:
            sma = df_with_indicators["close_10_sma"].dropna()
            assert len(sma) > 0

            # SMA should be less volatile than price
            price_vol = sample_data["close"].pct_change().std()
            sma_vol = sma.pct_change().std()
            assert sma_vol < price_vol

    def test_validation_against_reference(self, sample_data):
        """Test validation against reference implementations."""
        # This test requires the 'ta' library
        try:
            validation_results = validate_indicators_against_reference(sample_data)

            # Should return a dictionary
            assert isinstance(validation_results, dict)

            # If RSI validation was performed, check the result
            if "rsi_14" in validation_results:
                assert isinstance(validation_results["rsi_14"], bool)

        except ImportError:
            # Skip if 'ta' library is not available
            pytest.skip("ta library not available for validation")

    def test_data_quality_after_indicators(self, sample_data):
        """Test that data quality is maintained after adding indicators."""
        df_with_indicators = add_basic_indicators(sample_data)

        # Check for infinite values
        numeric_cols = df_with_indicators.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(
                df_with_indicators[col]
            ).any(), f"Infinite values found in {col}"

        # Check that we don't have too many NaN values
        nan_counts = df_with_indicators.isnull().sum()
        for col in numeric_cols:
            if col not in ["open", "high", "low", "close", "volume"]:
                # Allow some NaN values for indicators due to rolling calculations
                assert (
                    nan_counts[col] < len(df_with_indicators) * 0.5
                ), f"Too many NaN values in {col}"

    def test_indicator_consistency(self, sample_data):
        """Test that indicators are consistent across multiple runs."""
        df1 = add_basic_indicators(sample_data)
        df2 = add_basic_indicators(sample_data)

        # Results should be identical
        for col in df1.columns:
            if col in df2.columns:
                pd.testing.assert_series_equal(df1[col], df2[col], check_names=False)


class TestIndicatorEdgeCases:
    """Test edge cases for indicator calculations."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Should handle empty DataFrame gracefully
        result = add_basic_indicators(empty_df)
        assert isinstance(result, pd.DataFrame)

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        single_row = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        # Should not crash, but most indicators will be NaN
        df_with_indicators = add_basic_indicators(single_row)
        assert df_with_indicators.shape[0] == 1

    def test_missing_columns(self):
        """Test handling of missing OHLCV columns."""
        incomplete_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        with pytest.raises(Exception):
            add_basic_indicators(incomplete_df)

    def test_all_nan_data(self):
        """Test handling of all-NaN data."""
        nan_df = pd.DataFrame(
            {
                "open": [np.nan] * 10,
                "high": [np.nan] * 10,
                "low": [np.nan] * 10,
                "close": [np.nan] * 10,
                "volume": [np.nan] * 10,
            }
        )

        # Should handle NaN data gracefully
        result = add_basic_indicators(nan_df)
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])
