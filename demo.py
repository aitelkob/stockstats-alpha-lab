#!/usr/bin/env python3
"""
Quick demo script for StockStats Alpha Lab.

This script demonstrates the core functionality of the project:
1. Data loading
2. Feature engineering with stockstats
3. Signal evaluation
4. Strategy backtesting
5. Performance reporting

Run with: python demo.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data import DataLoader
from indicators import add_basic_indicators
from labeling import LabelingEngine
from backtest import BacktestEngine, StrategyBuilder
from plots import Plotter

def main():
    """Run the complete demo pipeline."""
    print("🚀 StockStats Alpha Lab - Quick Demo")
    print("=" * 50)
    
    # 1. Data Loading
    print("\n📊 Step 1: Loading Data")
    print("-" * 30)
    
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="1y")
    print(f"✅ Loaded AAPL data: {df.shape[0]} records from {df.index.min().date()} to {df.index.max().date()}")
    
    # 2. Feature Engineering
    print("\n🔧 Step 2: Feature Engineering with StockStats")
    print("-" * 30)
    
    df_with_indicators = add_basic_indicators(df)
    indicator_count = len(df_with_indicators.columns) - 5  # Subtract OHLCV columns
    print(f"✅ Added {indicator_count} technical indicators")
    print(f"   Key indicators: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, etc.")
    
    # 3. Signal Evaluation
    print("\n📈 Step 3: Signal Evaluation")
    print("-" * 30)
    
    labeler = LabelingEngine()
    forward_returns = labeler.forward_return_label(df_with_indicators, horizon=5)
    binary_labels = labeler.binary_classification_label(forward_returns)
    
    print(f"✅ Created forward return labels (5-day horizon)")
    print(f"   Positive return rate: {binary_labels.mean():.1%}")
    
    # 4. Strategy Creation
    print("\n🎯 Step 4: Strategy Creation")
    print("-" * 30)
    
    # RSI + Trend strategy
    rsi_signals = StrategyBuilder.rsi_trend_strategy(df_with_indicators)
    print(f"✅ RSI + Trend strategy: {len(rsi_signals[rsi_signals != 0])} signals")
    
    # MACD crossover strategy
    macd_signals = StrategyBuilder.macd_crossover_strategy(df_with_indicators)
    print(f"✅ MACD crossover strategy: {len(macd_signals[macd_signals != 0])} signals")
    
    # 5. Backtesting
    print("\n💰 Step 5: Strategy Backtesting")
    print("-" * 30)
    
    engine = BacktestEngine(initial_capital=100000)
    
    # Backtest RSI strategy
    rsi_results = engine.run_backtest(df_with_indicators, rsi_signals, strategy_name="RSI_Trend")
    print(f"✅ RSI Strategy Results:")
    print(f"   Total Return: {rsi_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {rsi_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {rsi_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {rsi_results['hit_rate']:.1%}")
    
    # Backtest MACD strategy
    macd_results = engine.run_backtest(df_with_indicators, macd_signals, strategy_name="MACD_Crossover")
    print(f"\n✅ MACD Strategy Results:")
    print(f"   Total Return: {macd_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {macd_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {macd_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {macd_results['hit_rate']:.1%}")
    
    # 6. Strategy Comparison
    print("\n📊 Step 6: Strategy Comparison")
    print("-" * 30)
    
    strategies = {
        'RSI_Trend': rsi_signals,
        'MACD_Crossover': macd_signals
    }
    
    from backtest import run_strategy_comparison
    comparison_df = run_strategy_comparison(df_with_indicators, strategies)
    print("✅ Strategy Comparison:")
    print(comparison_df[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown']].to_string(index=False))
    
    # 7. Information Coefficient Analysis
    print("\n🧠 Step 7: Information Coefficient Analysis")
    print("-" * 30)
    
    from labeling import calculate_information_coefficient
    
    # Calculate IC for key indicators
    indicator_cols = [col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    ic_values = calculate_information_coefficient(df_with_indicators[indicator_cols], forward_returns)
    
    print("✅ Top 5 Indicators by IC:")
    for indicator, ic in ic_values.head().items():
        print(f"   {indicator}: {ic:.3f}")
    
    # 8. Summary
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("This demo showcased:")
    print("• Clean data loading with yfinance")
    print("• One-line technical indicators with stockstats")
    print("• Time-series disciplined labeling")
    print("• Vectorized backtesting with realistic costs")
    print("• Comprehensive performance analysis")
    print("\nNext steps:")
    print("• Run: jupyter notebook notebooks/")
    print("• Explore: python -m pytest tests/")
    print("• Read: README.md for full documentation")
    
    return {
        'rsi_results': rsi_results,
        'macd_results': macd_results,
        'comparison': comparison_df,
        'ic_analysis': ic_values
    }

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have installed all dependencies:")
        print("pip install -e .")
        sys.exit(1)
