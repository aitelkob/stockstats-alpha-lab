"""
Labeling module for creating targets from financial time series.

This module implements various labeling schemes including forward returns,
triple-barrier labeling, and regime-based labels for machine learning.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelingEngine:
    """Main class for creating various types of labels from financial data."""

    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize labeling engine.

        Args:
            transaction_cost: Transaction cost as a fraction (e.g., 0.001 = 0.1%)
        """
        self.transaction_cost = transaction_cost

    def forward_return_label(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        target_col: str = "close",
        return_type: str = "log",
    ) -> pd.Series:
        """
        Create forward return labels with proper time-series discipline.

        This is the core labeling function mentioned in the project plan.

        Args:
            df: DataFrame with price data
            horizon: Number of periods ahead to look
            target_col: Column name for target prices
            return_type: Type of return ('log', 'simple', 'excess')

        Returns:
            Series with forward returns
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame")

        prices = df[target_col]

        if return_type == "log":
            # k-day forward log return (avoid peeking)
            forward_prices = prices.shift(-horizon)
            y = np.log(forward_prices / prices)
        elif return_type == "simple":
            forward_prices = prices.shift(-horizon)
            y = (forward_prices - prices) / prices
        elif return_type == "excess":
            # Excess return over risk-free rate (simplified)
            forward_prices = prices.shift(-horizon)
            simple_returns = (forward_prices - prices) / prices
            # Assume 2% annual risk-free rate
            risk_free_daily = 0.02 / 252
            y = simple_returns - (risk_free_daily * horizon)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

        logger.info(f"Created {return_type} forward returns with horizon {horizon}")
        return y

    def triple_barrier_label(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        volatility_col: str = "atr_14",
        horizon: int = 20,
        upper_barrier: float = 2.0,
        lower_barrier: float = 2.0,
        min_touch_days: int = 1,
    ) -> pd.Series:
        """
        Create triple-barrier labels for classification.

        Args:
            df: DataFrame with price and volatility data
            price_col: Column name for prices
            volatility_col: Column name for volatility measure
            horizon: Maximum holding period
            upper_barrier: Upper barrier as multiple of volatility
            lower_barrier: Lower barrier as multiple of volatility
            min_touch_days: Minimum days before barrier can be touched

        Returns:
            Series with labels: 1 (upper barrier), -1 (lower barrier), 0 (timeout)
        """
        if price_col not in df.columns:
            raise ValueError(f"Price column {price_col} not found")
        if volatility_col not in df.columns:
            raise ValueError(f"Volatility column {volatility_col} not found")

        prices = df[price_col]
        volatility = df[volatility_col]

        labels = pd.Series(0, index=prices.index)

        for i in range(len(prices) - horizon):
            if pd.isna(prices.iloc[i]) or pd.isna(volatility.iloc[i]):
                continue

            start_price = prices.iloc[i]
            vol = volatility.iloc[i]

            # Calculate barriers
            upper_threshold = start_price * (1 + upper_barrier * vol)
            lower_threshold = start_price * (1 - lower_barrier * vol)

            # Look forward for barrier touches
            future_prices = prices.iloc[i + min_touch_days : i + horizon + 1]

            if len(future_prices) == 0:
                continue

            # Check for upper barrier touch
            upper_touch = (future_prices >= upper_threshold).any()
            # Check for lower barrier touch
            lower_touch = (future_prices <= lower_threshold).any()

            if upper_touch and lower_touch:
                # Both barriers touched - use first one
                upper_idx = future_prices[future_prices >= upper_threshold].index[0]
                lower_idx = future_prices[future_prices <= lower_threshold].index[0]
                if upper_idx < lower_idx:
                    labels.iloc[i] = 1
                else:
                    labels.iloc[i] = -1
            elif upper_touch:
                labels.iloc[i] = 1
            elif lower_touch:
                labels.iloc[i] = -1
            # else remains 0 (timeout)

        logger.info(f"Created triple-barrier labels: {labels.value_counts().to_dict()}")
        return labels

    def regime_based_label(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        volatility_col: str = "atr_14",
        trend_col: str = "close_20_sma",
        regime_threshold: float = 0.5,
    ) -> pd.Series:
        """
        Create regime-based labels based on market conditions.

        Args:
            df: DataFrame with price, volatility, and trend data
            price_col: Column name for prices
            volatility_col: Column name for volatility measure
            trend_col: Column name for trend indicator
            regime_threshold: Threshold for regime classification

        Returns:
            Series with regime labels
        """
        if price_col not in df.columns:
            raise ValueError(f"Price column {price_col} not found")
        if volatility_col not in df.columns:
            raise ValueError(f"Volatility column {volatility_col} not found")
        if trend_col not in df.columns:
            raise ValueError(f"Trend column {trend_col} not found")

        prices = df[price_col]
        volatility = df[volatility_col]
        trend = df[trend_col]

        # Calculate rolling statistics for regime classification
        vol_ma = volatility.rolling(20).mean()
        vol_std = volatility.rolling(20).std()

        # High volatility regime
        high_vol = volatility > (vol_ma + regime_threshold * vol_std)

        # Trend regime
        uptrend = prices > trend

        # Create regime labels
        labels = pd.Series(0, index=prices.index)  # 0 = normal
        labels[high_vol & uptrend] = 1  # 1 = high vol uptrend
        labels[high_vol & ~uptrend] = 2  # 2 = high vol downtrend
        labels[~high_vol & uptrend] = 3  # 3 = low vol uptrend
        labels[~high_vol & ~uptrend] = 4  # 4 = low vol downtrend

        logger.info(f"Created regime labels: {labels.value_counts().to_dict()}")
        return labels

    def binary_classification_label(
        self,
        forward_returns: pd.Series,
        threshold: float = 0.0,
        transaction_cost: Optional[float] = None,
    ) -> pd.Series:
        """
        Create binary classification labels from forward returns.

        Args:
            forward_returns: Series of forward returns
            threshold: Return threshold for classification
            transaction_cost: Transaction cost to subtract from returns

        Returns:
            Series with binary labels: 1 (positive), 0 (negative)
        """
        if transaction_cost is None:
            transaction_cost = self.transaction_cost

        # Adjust returns for transaction costs
        adjusted_returns = forward_returns - transaction_cost

        # Create binary labels
        labels = (adjusted_returns > threshold).astype(int)

        logger.info(f"Created binary labels: {labels.value_counts().to_dict()}")
        return labels

    def multi_class_label(
        self,
        forward_returns: pd.Series,
        quantiles: List[float] = [0.25, 0.75],
        transaction_cost: Optional[float] = None,
    ) -> pd.Series:
        """
        Create multi-class labels from forward returns using quantiles.

        Args:
            forward_returns: Series of forward returns
            quantiles: Quantile thresholds for classification
            transaction_cost: Transaction cost to subtract from returns

        Returns:
            Series with multi-class labels
        """
        if transaction_cost is None:
            transaction_cost = self.transaction_cost

        # Adjust returns for transaction costs
        adjusted_returns = forward_returns - transaction_cost

        # Calculate quantile thresholds
        thresholds = adjusted_returns.quantile(quantiles)

        # Create multi-class labels
        labels = pd.Series(0, index=adjusted_returns.index)  # 0 = middle class

        for i, threshold in enumerate(thresholds):
            if i == 0:
                labels[adjusted_returns <= threshold] = 0  # Bottom class
            else:
                labels[adjusted_returns > threshold] = len(quantiles)  # Top class

        logger.info(f"Created multi-class labels: {labels.value_counts().to_dict()}")
        return labels


def create_feature_matrix(
    df: pd.DataFrame,
    indicator_columns: List[str],
    label_column: str,
    lookback_window: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix and labels for machine learning.

    Args:
        df: DataFrame with indicators and labels
        indicator_columns: List of indicator column names
        label_column: Name of label column
        lookback_window: Number of past periods to include as features

    Returns:
        Tuple of (feature_matrix, labels)
    """
    # Select indicator columns
    feature_cols = [col for col in indicator_columns if col in df.columns]

    if not feature_cols:
        raise ValueError("No valid indicator columns found")

    # Create feature matrix
    X = df[feature_cols].copy()

    # Add lookback features
    if lookback_window > 1:
        for col in feature_cols:
            for lag in range(1, lookback_window):
                X[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Get labels
    y = df[label_column].copy()

    # Remove rows with NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Created feature matrix: {X.shape}, labels: {len(y)}")
    return X, y


def calculate_information_coefficient(
    features: pd.DataFrame, labels: pd.Series, method: str = "spearman"
) -> pd.Series:
    """
    Calculate Information Coefficient (IC) between features and labels.

    Args:
        features: DataFrame of features
        labels: Series of labels
        method: Correlation method ('spearman', 'pearson', 'kendall')

    Returns:
        Series with IC values for each feature
    """
    ic_values = {}

    for col in features.columns:
        if method == "spearman":
            ic_values[col] = features[col].corr(labels, method="spearman")
        elif method == "pearson":
            ic_values[col] = features[col].corr(labels, method="pearson")
        elif method == "kendall":
            ic_values[col] = features[col].corr(labels, method="kendall")
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    ic_series = pd.Series(ic_values)
    ic_series = ic_series.dropna().sort_values(ascending=False)

    logger.info(f"Calculated IC for {len(ic_series)} features")
    return ic_series


if __name__ == "__main__":
    # Example usage
    from data import DataLoader
    from indicators import add_basic_indicators

    # Load and prepare data
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="1y")
    df = add_basic_indicators(df)

    # Create labeling engine
    labeler = LabelingEngine()

    # Create forward return labels
    forward_returns = labeler.forward_return_label(df, horizon=5)
    print(f"Forward returns created: {forward_returns.dropna().shape}")

    # Create binary classification labels
    binary_labels = labeler.binary_classification_label(forward_returns)
    print(f"Binary labels: {binary_labels.value_counts().to_dict()}")

    # Create triple-barrier labels
    triple_barrier_labels = labeler.triple_barrier_label(df)
    print(f"Triple-barrier labels: {triple_barrier_labels.value_counts().to_dict()}")

    # Calculate IC
    indicator_cols = [
        col
        for col in df.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]
    ic_values = calculate_information_coefficient(df[indicator_cols], forward_returns)
    print(f"Top 5 IC values:\n{ic_values.head()}")
