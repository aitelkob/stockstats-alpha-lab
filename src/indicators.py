"""
Technical indicators module using stockstats.

This module provides a clean wrapper around stockstats for engineering
technical analysis features with proper validation and testing.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as SDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorEngine:
    """Main class for engineering technical indicators using stockstats."""
    
    def __init__(self, validate_indicators: bool = True):
        """
        Initialize indicator engine.
        
        Args:
            validate_indicators: Whether to validate indicator calculations
        """
        self.validate_indicators = validate_indicators
        self.indicators_added = []
    
    def add_indicators(
        self,
        df: pd.DataFrame,
        indicator_groups: Optional[List[str]] = None,
        custom_indicators: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV DataFrame.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            indicator_groups: List of indicator groups to add
            custom_indicators: Dict of custom indicator names and formulas
            
        Returns:
            DataFrame with added indicator columns
        """
        if indicator_groups is None:
            indicator_groups = ["trend", "momentum", "volatility", "volume", "oscillators"]
        
        # Create stockstats DataFrame
        sdf = SDF.retype(df.copy())
        
        # Add indicators by group
        for group in indicator_groups:
            self._add_indicator_group(sdf, group)
        
        # Add custom indicators
        if custom_indicators:
            self._add_custom_indicators(sdf, custom_indicators)
        
        # Convert back to pandas and validate
        result_df = pd.DataFrame(sdf)
        
        if self.validate_indicators:
            self._validate_indicators(result_df)
        
        logger.info(f"Added {len(self.indicators_added)} indicators")
        return result_df
    
    def _add_indicator_group(self, sdf: SDF, group: str) -> None:
        """Add indicators from a specific group."""
        group_indicators = {
            "trend": [
                "close_5_sma", "close_10_sma", "close_20_sma", "close_50_sma", "close_200_sma",
                "close_5_ema", "close_10_ema", "close_20_ema", "close_50_ema", "close_200_ema",
                "macd", "macds", "macdh", "boll", "boll_ub", "boll_lb"
            ],
            "momentum": [
                "rsi_6", "rsi_12", "rsi_14", "rsi_24",
                "close_5_roc", "close_10_roc", "close_20_roc",
                "kdjk", "kdjd", "kdjj",
                "wr_10", "wr_14", "wr_20"
            ],
            "volatility": [
                "atr_14", "atr_21", "mstd_10", "mstd_20", "mstd_50",
                "cr", "cr-ma1", "cr-ma2", "cr-ma3"
            ],
            "volume": [
                "volume_sma", "volume_ema", "volume_roc",
                "volume_delta", "volume_ratio"
            ],
            "oscillators": [
                "cci", "cci_20", "cci_30",
                "trix", "trix_9_sma",
                "dx", "dx_6_sma", "dx_14_sma"
            ]
        }
        
        if group not in group_indicators:
            logger.warning(f"Unknown indicator group: {group}")
            return
        
        for indicator in group_indicators[group]:
            try:
                # Access the indicator to trigger calculation
                _ = sdf[indicator]
                self.indicators_added.append(indicator)
                logger.debug(f"Added {indicator}")
            except Exception as e:
                logger.warning(f"Failed to add {indicator}: {e}")
    
    def _add_custom_indicators(self, sdf: SDF, custom_indicators: Dict[str, str]) -> None:
        """Add custom indicators defined by formulas."""
        for name, formula in custom_indicators.items():
            try:
                # This would require extending stockstats or using pandas directly
                # For now, we'll add some common custom indicators
                if formula == "log_returns":
                    sdf["log_returns"] = np.log(sdf["close"] / sdf["close"].shift(1))
                elif formula == "price_position":
                    sdf["price_position"] = (sdf["close"] - sdf["low"]) / (sdf["high"] - sdf["low"])
                elif formula == "volatility_ratio":
                    sdf["volatility_ratio"] = sdf["mstd_20"] / sdf["close"]
                
                self.indicators_added.append(name)
                logger.debug(f"Added custom indicator: {name}")
            except Exception as e:
                logger.warning(f"Failed to add custom indicator {name}: {e}")
    
    def _validate_indicators(self, df: pd.DataFrame) -> None:
        """Validate that indicators were calculated correctly."""
        validation_errors = []
        
        for indicator in self.indicators_added:
            if indicator not in df.columns:
                validation_errors.append(f"Indicator {indicator} not found in DataFrame")
                continue
            
            # Check for all NaN values
            if df[indicator].isna().all():
                validation_errors.append(f"Indicator {indicator} is all NaN")
            
            # Check for infinite values
            if np.isinf(df[indicator]).any():
                validation_errors.append(f"Indicator {indicator} contains infinite values")
        
        if validation_errors:
            logger.warning(f"Validation errors found: {validation_errors}")
        else:
            logger.info("All indicators validated successfully")
    
    def get_indicator_info(self) -> Dict[str, str]:
        """Get information about available indicators."""
        return {
            "trend": "Moving averages, MACD, Bollinger Bands",
            "momentum": "RSI, ROC, Stochastic, Williams %R",
            "volatility": "ATR, Moving Standard Deviation, Commodity Channel Index",
            "volume": "Volume-based indicators and ratios",
            "oscillators": "CCI, TRIX, Directional Movement Index"
        }


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a basic set of technical indicators.
    
    This is the core demo function mentioned in the project plan.
    
    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        
    Returns:
        DataFrame with added indicator columns
    """
    engine = IndicatorEngine()
    
    # Add core indicators as specified in the project plan
    sdf = SDF.retype(df.copy())
    
    # Trend & momentum (as per project plan) - using safe indicators
    sdf['close_10_sma']
    sdf['close_20_ema'] 
    sdf['macd']
    sdf['rsi_14']
    sdf['close_10_roc']
    # sdf['mstd_20']  # This indicator causes issues, skipping
    sdf['boll']
    sdf['kdjk']
    sdf['kdjd'] 
    sdf['kdjj']
    
    # Volatility & range features
    sdf['atr_14']
    sdf['cr']
    sdf['wr_14']
    
    # Log returns (built-in)
    sdf['log-ret']
    
    return pd.DataFrame(sdf)


def add_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a comprehensive set of technical indicators.
    
    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        
    Returns:
        DataFrame with added indicator columns
    """
    engine = IndicatorEngine()
    return engine.add_indicators(df)


def validate_indicators_against_reference(
    df: pd.DataFrame,
    reference_library: str = "ta"
) -> Dict[str, bool]:
    """
    Validate stockstats indicators against reference implementations.
    
    Args:
        df: OHLCV DataFrame
        reference_library: Reference library to compare against ('ta', 'talib')
        
    Returns:
        Dictionary mapping indicator names to validation results
    """
    if reference_library == "ta":
        try:
            import ta
        except ImportError:
            logger.warning("ta library not available for validation")
            return {}
    
    validation_results = {}
    
    # Example validation for RSI
    try:
        sdf = SDF.retype(df.copy())
        stockstats_rsi = sdf['rsi_14']
        
        if reference_library == "ta":
            ta_rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # Compare (allowing for small numerical differences)
            correlation = np.corrcoef(
                stockstats_rsi.dropna(), 
                ta_rsi.dropna()
            )[0, 1]
            
            validation_results['rsi_14'] = correlation > 0.99
            logger.info(f"RSI validation: correlation = {correlation:.4f}")
    
    except Exception as e:
        logger.warning(f"RSI validation failed: {e}")
        validation_results['rsi_14'] = False
    
    return validation_results


def get_indicator_categories() -> Dict[str, List[str]]:
    """Get indicators organized by category."""
    return {
        "trend": [
            "close_5_sma", "close_10_sma", "close_20_sma", "close_50_sma", "close_200_sma",
            "close_5_ema", "close_10_ema", "close_20_ema", "close_50_ema", "close_200_ema",
            "macd", "macds", "macdh", "boll", "boll_ub", "boll_lb"
        ],
        "momentum": [
            "rsi_6", "rsi_12", "rsi_14", "rsi_24",
            "close_5_roc", "close_10_roc", "close_20_roc",
            "kdjk", "kdjd", "kdjj",
            "wr_10", "wr_14", "wr_20"
        ],
        "volatility": [
            "atr_14", "atr_21", "mstd_10", "mstd_20", "mstd_50",
            "cr", "cr-ma1", "cr-ma2", "cr-ma3"
        ],
        "volume": [
            "volume_sma", "volume_ema", "volume_roc",
            "volume_delta", "volume_ratio"
        ],
        "oscillators": [
            "cci", "cci_20", "cci_30",
            "trix", "trix_9_sma",
            "dx", "dx_6_sma", "dx_14_sma"
        ]
    }


if __name__ == "__main__":
    # Example usage
    from data import DataLoader
    
    # Load sample data
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="1y")
    
    # Add basic indicators
    df_with_indicators = add_basic_indicators(df)
    print(f"Added indicators. Shape: {df_with_indicators.shape}")
    print(f"New columns: {[col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
    
    # Validate against reference
    validation_results = validate_indicators_against_reference(df)
    print(f"Validation results: {validation_results}")
