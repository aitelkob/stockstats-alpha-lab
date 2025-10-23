"""
StockStats Alpha Lab - A comprehensive quantitative finance research platform.

This package provides tools for:
- Data loading and quality assessment
- Technical indicator engineering with stockstats
- Signal evaluation and cross-validation
- Strategy backtesting with realistic costs
- Risk analysis and reporting

Key modules:
- data: OHLCV data loading and validation
- indicators: Technical indicator engineering
- labeling: Forward return and triple-barrier labeling
- models: Machine learning pipelines
- backtest: Vectorized backtesting engine
- plots: Visualization and reporting
"""

__version__ = "0.1.0"
__author__ = "Data Scientist"
__email__ = "candidate@example.com"

# Import main classes for easy access
from .data import DataLoader, get_sample_tickers, validate_data_quality
from .indicators import add_basic_indicators, add_comprehensive_indicators, IndicatorEngine
from .labeling import LabelingEngine, create_feature_matrix, calculate_information_coefficient
from .models import ModelPipeline, create_baseline_models, compare_models
from .backtest import BacktestEngine, StrategyBuilder, run_strategy_comparison
from .plots import Plotter

__all__ = [
    # Data
    "DataLoader",
    "get_sample_tickers", 
    "validate_data_quality",
    
    # Indicators
    "add_basic_indicators",
    "add_comprehensive_indicators",
    "IndicatorEngine",
    
    # Labeling
    "LabelingEngine",
    "create_feature_matrix",
    "calculate_information_coefficient",
    
    # Models
    "ModelPipeline",
    "create_baseline_models",
    "compare_models",
    
    # Backtesting
    "BacktestEngine",
    "StrategyBuilder",
    "run_strategy_comparison",
    
    # Plotting
    "Plotter",
]
