"""
Advanced Technical Indicators Module

This module provides sophisticated technical indicators and custom formulas
for quantitative finance analysis, including momentum, volatility, and trend indicators.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from stockstats import StockDataFrame as SDF

logger = logging.getLogger(__name__)


class AdvancedIndicatorEngine:
    """Advanced indicator engine for sophisticated technical analysis."""
    
    def __init__(self):
        """Initialize the advanced indicator engine."""
        self.indicators = {}
        self.custom_formulas = {}
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced momentum indicators."""
        sdf = SDF.retype(df.copy())
        
        # Rate of Change variations
        sdf['roc_5']      # 5-period ROC
        sdf['roc_10']     # 10-period ROC
        sdf['roc_20']     # 20-period ROC
        
        # Momentum indicators
        sdf['mom_10']     # 10-period momentum
        sdf['mom_20']     # 20-period momentum
        
        # Williams %R
        sdf['wr_14']      # 14-period Williams %R
        sdf['wr_21']      # 21-period Williams %R
        
        # Stochastic Oscillator
        sdf['kdjk']       # %K
        sdf['kdjd']       # %D
        sdf['kdjj']       # %J
        
        # Commodity Channel Index
        sdf['cci_14']     # 14-period CCI
        sdf['cci_20']     # 20-period CCI
        
        # Money Flow Index
        sdf['mfi_14']     # 14-period MFI
        
        return pd.DataFrame(sdf)
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility indicators."""
        sdf = SDF.retype(df.copy())
        
        # Average True Range variations
        sdf['atr_14']     # 14-period ATR
        sdf['atr_21']     # 21-period ATR
        
        # Bollinger Bands variations
        sdf['boll']       # Bollinger Bands
        sdf['boll_ub']    # Upper Band
        sdf['boll_lb']    # Lower Band
        sdf['boll_wb']    # Band Width
        sdf['boll_pb']    # %B
        
        # Keltner Channels
        sdf['kc_20']      # 20-period Keltner Channel
        sdf['kc_ub']      # Upper Keltner
        sdf['kc_lb']      # Lower Keltner
        
        # Donchian Channels
        sdf['dc_20']      # 20-period Donchian
        sdf['dc_ub']      # Upper Donchian
        sdf['dc_lb']      # Lower Donchian
        
        # Historical Volatility
        sdf['hv_10']      # 10-period HV
        sdf['hv_20']      # 20-period HV
        sdf['hv_30']      # 30-period HV
        
        return pd.DataFrame(sdf)
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced trend indicators."""
        sdf = SDF.retype(df.copy())
        
        # Moving Average variations
        sdf['close_5_sma']    # 5-period SMA
        sdf['close_10_sma']   # 10-period SMA
        sdf['close_20_sma']   # 20-period SMA
        sdf['close_50_sma']   # 50-period SMA
        sdf['close_100_sma']  # 100-period SMA
        sdf['close_200_sma']  # 200-period SMA
        
        sdf['close_5_ema']    # 5-period EMA
        sdf['close_10_ema']   # 10-period EMA
        sdf['close_20_ema']   # 20-period EMA
        sdf['close_50_ema']   # 50-period EMA
        
        # MACD variations
        sdf['macd']           # MACD Line
        sdf['macds']          # Signal Line
        sdf['macdh']          # Histogram
        
        # Parabolic SAR
        sdf['psar']           # Parabolic SAR
        
        # Ichimoku Cloud
        sdf['ichimoku_a']     # Ichimoku A
        sdf['ichimoku_b']     # Ichimoku B
        sdf['ichimoku_base']  # Base Line
        sdf['ichimoku_con']   # Conversion Line
        
        # ADX (Average Directional Index)
        sdf['adx_14']         # 14-period ADX
        sdf['di_plus']        # +DI
        sdf['di_minus']       # -DI
        
        return pd.DataFrame(sdf)
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        sdf = SDF.retype(df.copy())
        
        # On-Balance Volume
        sdf['obv']            # On-Balance Volume
        
        # Volume Rate of Change
        sdf['vroc_5']         # 5-period VROC
        sdf['vroc_10']        # 10-period VROC
        
        # Volume Moving Average
        sdf['vma_5']          # 5-period VMA
        sdf['vma_10']         # 10-period VMA
        sdf['vma_20']         # 20-period VMA
        
        # Accumulation/Distribution Line
        sdf['ad']             # A/D Line
        
        # Chaikin Money Flow
        sdf['cmf_20']         # 20-period CMF
        
        # Volume Price Trend
        sdf['vpt']            # Volume Price Trend
        
        return pd.DataFrame(sdf)
    
    def add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom calculated indicators."""
        result_df = df.copy()
        
        # Price Position (relative position within recent range)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            lookback = 20
            high_max = df['high'].rolling(window=lookback).max()
            low_min = df['low'].rolling(window=lookback).min()
            result_df['price_position'] = (df['close'] - low_min) / (high_max - low_min)
        
        # Volatility Ratio (current vs historical volatility)
        if 'close' in df.columns:
            short_vol = df['close'].pct_change().rolling(10).std()
            long_vol = df['close'].pct_change().rolling(30).std()
            result_df['volatility_ratio'] = short_vol / long_vol
        
        # Price Acceleration (second derivative of price)
        if 'close' in df.columns:
            price_velocity = df['close'].diff()
            result_df['price_acceleration'] = price_velocity.diff()
        
        # Volume-Price Divergence
        if all(col in df.columns for col in ['close', 'volume']):
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            result_df['volume_price_divergence'] = price_change - volume_change
        
        # Trend Strength (ADX-like calculation)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            result_df['trend_strength'] = tr.rolling(14).mean()
        
        # Support/Resistance Levels
        if 'close' in df.columns:
            window = 20
            highs = df['high'].rolling(window=window, center=True).max()
            lows = df['low'].rolling(window=window, center=True).min()
            
            result_df['resistance_level'] = highs
            result_df['support_level'] = lows
            result_df['price_vs_resistance'] = df['close'] / highs
            result_df['price_vs_support'] = df['close'] / lows
        
        # Market Regime Indicators
        if 'close' in df.columns:
            # Bull/Bear market indicator
            sma_50 = df['close'].rolling(50).mean()
            sma_200 = df['close'].rolling(200).mean()
            result_df['market_regime'] = np.where(
                sma_50 > sma_200, 1,  # Bull market
                np.where(sma_50 < sma_200, -1, 0)  # Bear market or neutral
            )
            
            # Volatility regime
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_50 = df['close'].pct_change().rolling(50).std()
            result_df['volatility_regime'] = np.where(
                vol_20 > vol_50 * 1.2, 1,  # High volatility
                np.where(vol_20 < vol_50 * 0.8, -1, 0)  # Low volatility or normal
            )
        
        return result_df
    
    def add_fractal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fractal and pattern recognition indicators."""
        result_df = df.copy()
        
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return result_df
        
        # Fractal Highs and Lows
        window = 5
        highs = df['high']
        lows = df['low']
        
        # Fractal high: high is higher than surrounding highs
        fractal_high = (highs == highs.rolling(window=window, center=True).max()) & \
                      (highs > highs.shift(1)) & (highs > highs.shift(-1))
        
        # Fractal low: low is lower than surrounding lows
        fractal_low = (lows == lows.rolling(window=window, center=True).min()) & \
                     (lows < lows.shift(1)) & (lows < lows.shift(-1))
        
        result_df['fractal_high'] = fractal_high.astype(int)
        result_df['fractal_low'] = fractal_low.astype(int)
        
        # Pivot Points
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        # Standard Pivot Points
        result_df['pivot_point'] = (prev_high + prev_low + prev_close) / 3
        result_df['resistance_1'] = 2 * result_df['pivot_point'] - prev_low
        result_df['support_1'] = 2 * result_df['pivot_point'] - prev_high
        result_df['resistance_2'] = result_df['pivot_point'] + (prev_high - prev_low)
        result_df['support_2'] = result_df['pivot_point'] - (prev_high - prev_low)
        
        # Fibonacci Retracements
        swing_high = df['high'].rolling(20).max()
        swing_low = df['low'].rolling(20).min()
        swing_range = swing_high - swing_low
        
        result_df['fib_23.6'] = swing_high - 0.236 * swing_range
        result_df['fib_38.2'] = swing_high - 0.382 * swing_range
        result_df['fib_50.0'] = swing_high - 0.500 * swing_range
        result_df['fib_61.8'] = swing_high - 0.618 * swing_range
        result_df['fib_78.6'] = swing_high - 0.786 * swing_range
        
        return result_df
    
    def add_cyclical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical and seasonal indicators."""
        result_df = df.copy()
        
        if 'close' in df.columns:
            # Detrended Price Oscillator
            period = 20
            sma = df['close'].rolling(period).mean()
            result_df['dpo'] = df['close'].shift(period // 2 + 1) - sma
            
            # Hilbert Transform Dominant Cycle Period
            # Simplified version using price cycles
            price_cycles = df['close'].rolling(20).apply(
                lambda x: len(np.where(np.diff(np.sign(np.diff(x))))[0]) if len(x) > 2 else 0
            )
            result_df['cycle_period'] = price_cycles
            
            # Seasonal indicators (if we have enough data)
            if len(df) > 252:  # More than a year
                df_with_dates = df.copy()
                if hasattr(df.index, 'month'):
                    result_df['month'] = df.index.month
                    result_df['quarter'] = ((df.index.month - 1) // 3) + 1
                    result_df['day_of_week'] = df.index.dayofweek
                    result_df['day_of_month'] = df.index.day
                
                # Monthly seasonality
                monthly_returns = df['close'].pct_change().groupby(df.index.month).mean()
                result_df['monthly_seasonality'] = df.index.month.map(monthly_returns)
        
        return result_df
    
    def add_market_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure indicators."""
        result_df = df.copy()
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return result_df
        
        # Bid-Ask Spread Proxy (using OHLC)
        result_df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price Impact (volume-weighted price change)
        price_change = df['close'].pct_change()
        result_df['price_impact'] = price_change / df['volume'].rolling(5).mean()
        
        # Order Flow Imbalance
        up_volume = df['volume'].where(df['close'] > df['open'], 0)
        down_volume = df['volume'].where(df['close'] < df['open'], 0)
        result_df['order_flow_imbalance'] = (up_volume - down_volume) / (up_volume + down_volume)
        
        # Volume-Price Trend
        result_df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # Ease of Movement
        distance_moved = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        box_height = df['volume'] / (df['high'] - df['low'])
        result_df['ease_of_movement'] = distance_moved / box_height
        
        # Force Index
        result_df['force_index'] = df['close'].diff() * df['volume']
        result_df['force_index_ema'] = result_df['force_index'].ewm(span=13).mean()
        
        return result_df
    
    def add_all_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all advanced indicators to the DataFrame."""
        logger.info("Adding advanced technical indicators...")
        
        # Start with basic indicators
        result_df = df.copy()
        
        # Add momentum indicators
        result_df = self.add_momentum_indicators(result_df)
        
        # Add volatility indicators
        result_df = self.add_volatility_indicators(result_df)
        
        # Add trend indicators
        result_df = self.add_trend_indicators(result_df)
        
        # Add volume indicators
        result_df = self.add_volume_indicators(result_df)
        
        # Add custom indicators
        result_df = self.add_custom_indicators(result_df)
        
        # Add fractal indicators
        result_df = self.add_fractal_indicators(result_df)
        
        # Add cyclical indicators
        result_df = self.add_cyclical_indicators(result_df)
        
        # Add microstructure indicators
        result_df = self.add_market_microstructure_indicators(result_df)
        
        logger.info(f"Added {len(result_df.columns) - len(df.columns)} advanced indicators")
        return result_df
    
    def get_indicator_categories(self) -> Dict[str, List[str]]:
        """Get indicators organized by category."""
        return {
            'momentum': [
                'roc_5', 'roc_10', 'roc_20', 'mom_10', 'mom_20',
                'wr_14', 'wr_21', 'kdjk', 'kdjd', 'kdjj',
                'cci_14', 'cci_20', 'mfi_14'
            ],
            'volatility': [
                'atr_14', 'atr_21', 'boll', 'boll_ub', 'boll_lb',
                'boll_wb', 'boll_pb', 'kc_20', 'kc_ub', 'kc_lb',
                'dc_20', 'dc_ub', 'dc_lb', 'hv_10', 'hv_20', 'hv_30'
            ],
            'trend': [
                'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_50_sma',
                'close_100_sma', 'close_200_sma', 'close_5_ema', 'close_10_ema',
                'close_20_ema', 'close_50_ema', 'macd', 'macds', 'macdh',
                'psar', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base',
                'ichimoku_con', 'adx_14', 'di_plus', 'di_minus'
            ],
            'volume': [
                'obv', 'vroc_5', 'vroc_10', 'vma_5', 'vma_10', 'vma_20',
                'ad', 'cmf_20', 'vpt'
            ],
            'custom': [
                'price_position', 'volatility_ratio', 'price_acceleration',
                'volume_price_divergence', 'trend_strength', 'resistance_level',
                'support_level', 'price_vs_resistance', 'price_vs_support',
                'market_regime', 'volatility_regime'
            ],
            'fractal': [
                'fractal_high', 'fractal_low', 'pivot_point', 'resistance_1',
                'support_1', 'resistance_2', 'support_2', 'fib_23.6', 'fib_38.2',
                'fib_50.0', 'fib_61.8', 'fib_78.6'
            ],
            'cyclical': [
                'dpo', 'cycle_period', 'month', 'quarter', 'day_of_week',
                'day_of_month', 'monthly_seasonality'
            ],
            'microstructure': [
                'spread_proxy', 'price_impact', 'order_flow_imbalance',
                'vpt', 'ease_of_movement', 'force_index', 'force_index_ema'
            ]
        }


def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all advanced technical indicators to a DataFrame.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with added advanced indicators
    """
    engine = AdvancedIndicatorEngine()
    return engine.add_all_advanced_indicators(df)


def create_custom_indicator(df: pd.DataFrame, formula: str, name: str) -> pd.Series:
    """
    Create a custom indicator using a formula.
    
    Args:
        df: OHLCV DataFrame
        formula: Formula string (e.g., 'close - close.shift(1)')
        name: Name for the indicator
        
    Returns:
        Series with the custom indicator
    """
    try:
        # Create a safe evaluation environment
        safe_dict = {
            'df': df,
            'close': df['close'],
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'volume': df['volume'],
            'np': np,
            'pd': pd
        }
        
        # Evaluate the formula
        result = eval(formula, {"__builtins__": {}}, safe_dict)
        
        if isinstance(result, pd.Series):
            result.name = name
            return result
        else:
            raise ValueError("Formula must return a pandas Series")
            
    except Exception as e:
        logger.error(f"Error creating custom indicator '{name}': {e}")
        return pd.Series(index=df.index, name=name, dtype=float)


def calculate_indicator_correlation(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified indicators.
    
    Args:
        df: DataFrame with indicators
        indicators: List of indicator column names
        
    Returns:
        Correlation matrix
    """
    available_indicators = [ind for ind in indicators if ind in df.columns]
    
    if not available_indicators:
        return pd.DataFrame()
    
    return df[available_indicators].corr()


def find_highly_correlated_indicators(df: pd.DataFrame, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    """
    Find pairs of highly correlated indicators.
    
    Args:
        df: DataFrame with indicators
        threshold: Correlation threshold
        
    Returns:
        List of (indicator1, indicator2, correlation) tuples
    """
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    indicator_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    if len(indicator_cols) < 2:
        return []
    
    corr_matrix = df[indicator_cols].corr()
    
    # Find highly correlated pairs
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                highly_correlated.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
    
    return sorted(highly_correlated, key=lambda x: abs(x[2]), reverse=True)
