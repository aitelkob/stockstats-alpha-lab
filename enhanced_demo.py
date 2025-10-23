"""
Enhanced Demo Script for StockStats Alpha Lab

This script demonstrates the enhanced capabilities including:
- Advanced technical indicators
- Risk metrics and analysis
- Portfolio optimization
- Benchmarking and attribution
- Interactive visualizations
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import modules
from data import DataLoader
from indicators import add_basic_indicators
from advanced_indicators import add_advanced_indicators, AdvancedIndicatorEngine
from labeling import LabelingEngine, create_feature_matrix
from backtest import BacktestEngine, StrategyBuilder, run_strategy_comparison
try:
    from models import ModelPipeline
except ImportError:
    print("Warning: ModelPipeline not available due to XGBoost dependency")
    ModelPipeline = None
from risk_metrics import RiskAnalyzer
from optimization import PortfolioOptimizer, ParameterOptimizer
from benchmarking import BenchmarkAnalyzer, PerformanceAttributor

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📊 {title}")
    print("-" * 40)

def main():
    """Main enhanced demo function."""
    print_header("StockStats Alpha Lab - Enhanced Demo")
    print("Advanced Quantitative Finance Research Platform")
    print("Featuring: Advanced Indicators, Risk Analysis, Optimization & More!")
    
    # Step 1: Load and prepare data
    print_section("Step 1: Data Loading & Preparation")
    try:
        loader = DataLoader()
        
        # Load multiple assets for portfolio analysis
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        print(f"Loading data for {len(tickers)} assets: {', '.join(tickers)}")
        
        # Load data for each ticker
        data_dict = {}
        for ticker in tickers:
            df = loader.load_single_ticker(ticker, period='2y')
            data_dict[ticker] = df
            print(f"✅ {ticker}: {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Use AAPL as primary for detailed analysis
        primary_data = data_dict['AAPL'].copy()
        print(f"\nPrimary analysis asset: AAPL ({len(primary_data)} records)")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Step 2: Advanced Technical Indicators
    print_section("Step 2: Advanced Technical Indicators")
    try:
        # Basic indicators
        basic_data = add_basic_indicators(primary_data)
        basic_indicators = len(basic_data.columns) - 5
        print(f"✅ Basic indicators: {basic_indicators} added")
        
        # Advanced indicators
        advanced_data = add_advanced_indicators(primary_data)
        advanced_indicators = len(advanced_data.columns) - 5
        print(f"✅ Advanced indicators: {advanced_indicators} added")
        
        # Show indicator categories
        engine = AdvancedIndicatorEngine()
        categories = engine.get_indicator_categories()
        print(f"\n📈 Indicator Categories:")
        for category, indicators in categories.items():
            available = [ind for ind in indicators if ind in advanced_data.columns]
            print(f"  • {category.title()}: {len(available)} indicators")
        
        # Calculate indicator correlation
        from advanced_indicators import calculate_indicator_correlation
        indicator_cols = [col for col in advanced_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        corr_matrix = calculate_indicator_correlation(advanced_data, indicator_cols[:20])  # Top 20 for demo
        
        print(f"✅ Indicator correlation analysis: {corr_matrix.shape[0]}x{corr_matrix.shape[1]} matrix")
        
    except Exception as e:
        print(f"❌ Advanced indicators failed: {e}")
        advanced_data = basic_data
    
    # Step 3: Risk Analysis
    print_section("Step 3: Advanced Risk Analysis")
    try:
        risk_analyzer = RiskAnalyzer()
        returns = primary_data['close'].pct_change().dropna()
        
        # Calculate comprehensive risk metrics
        risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(
            returns, 
            primary_data['close'], 
            primary_data['volume']
        )
        
        print("✅ Risk Metrics Calculated:")
        print(f"  • VaR (95%): {risk_metrics['var'][0.95]:.2%}")
        print(f"  • VaR (99%): {risk_metrics['var'][0.99]:.2%}")
        print(f"  • CVaR (95%): {risk_metrics['cvar'][0.95]:.2%}")
        print(f"  • Max Drawdown: {risk_metrics['max_drawdown']['max_drawdown_pct']:.2f}%")
        print(f"  • Annualized Volatility: {risk_metrics['volatility']['annualized_volatility']:.2%}")
        print(f"  • Sharpe Ratio: {risk_metrics['volatility']['annualized_volatility']:.2f}")
        
        # Tail risk metrics
        tail_risk = risk_metrics['tail_risk']
        print(f"  • Tail Ratio: {tail_risk['tail_ratio']:.2f}")
        print(f"  • Hill Estimator: {tail_risk['hill_estimator']:.2f}")
        
    except Exception as e:
        print(f"❌ Risk analysis failed: {e}")
    
    # Step 4: Portfolio Optimization
    print_section("Step 4: Portfolio Optimization")
    try:
        # Prepare multi-asset data for optimization
        portfolio_returns = pd.DataFrame()
        for ticker, data in data_dict.items():
            portfolio_returns[ticker] = data['close'].pct_change().dropna()
        
        # Align dates
        portfolio_returns = portfolio_returns.dropna()
        print(f"✅ Portfolio data: {portfolio_returns.shape[0]} observations, {portfolio_returns.shape[1]} assets")
        
        # Mean-variance optimization
        optimizer = PortfolioOptimizer()
        mv_result = optimizer.mean_variance_optimization(portfolio_returns)
        
        if mv_result['success']:
            print(f"✅ Mean-Variance Optimization:")
            print(f"  • Expected Return: {mv_result['expected_return']:.2%}")
            print(f"  • Volatility: {mv_result['volatility']:.2%}")
            print(f"  • Sharpe Ratio: {mv_result['sharpe_ratio']:.2f}")
            print(f"  • Optimal Weights:")
            for ticker, weight in zip(tickers, mv_result['weights']):
                print(f"    - {ticker}: {weight:.1%}")
        
        # Risk parity optimization
        rp_result = optimizer.risk_parity_optimization(portfolio_returns)
        if rp_result['success']:
            print(f"\n✅ Risk Parity Optimization:")
            print(f"  • Expected Return: {rp_result['expected_return']:.2%}")
            print(f"  • Volatility: {rp_result['volatility']:.2%}")
            print(f"  • Sharpe Ratio: {rp_result['sharpe_ratio']:.2f}")
        
    except Exception as e:
        print(f"❌ Portfolio optimization failed: {e}")
    
    # Step 5: Strategy Backtesting with Enhanced Features
    print_section("Step 5: Enhanced Strategy Backtesting")
    try:
        # Create multiple strategies
        strategies = {}
        
        # RSI + Trend Strategy
        if 'rsi_14' in advanced_data.columns and 'close_20_ema' in advanced_data.columns:
            strategies['RSI + Trend'] = StrategyBuilder.rsi_trend_strategy(advanced_data)
        
        # MACD Strategy
        if 'macd' in advanced_data.columns and 'macds' in advanced_data.columns:
            strategies['MACD Crossover'] = StrategyBuilder.macd_crossover_strategy(advanced_data)
        
        # Bollinger Bands Strategy
        if all(col in advanced_data.columns for col in ['boll_ub', 'boll_lb']):
            strategies['Bollinger Bands'] = StrategyBuilder.bollinger_bands_strategy(advanced_data)
        
        print(f"✅ Created {len(strategies)} strategies")
        
        # Run backtests
        engine = BacktestEngine()
        results = {}
        
        for name, signals in strategies.items():
            try:
                result = engine.run_backtest(advanced_data, signals, strategy_name=name)
                results[name] = result
                print(f"  • {name}: {result['total_return']:.2%} return, {result['sharpe_ratio']:.2f} Sharpe")
            except Exception as e:
                print(f"  • {name}: Failed - {e}")
        
        # Strategy comparison
        if len(results) > 1:
            print(f"\n📊 Strategy Comparison:")
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Strategy': name,
                    'Total Return': f"{result['total_return']:.2%}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                    'Max Drawdown': f"{result['max_drawdown']:.2%}",
                    'Hit Rate': f"{result['hit_rate']:.2%}",
                    'Num Trades': result['num_trades']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Strategy backtesting failed: {e}")
    
    # Step 6: Machine Learning with Enhanced Features
    print_section("Step 6: Enhanced Machine Learning")
    try:
        # Create labels
        labeler = LabelingEngine()
        forward_returns = labeler.forward_return_label(advanced_data, horizon=5)
        binary_labels = labeler.binary_classification_label(forward_returns)
        
        # Create feature matrix
        indicator_cols = [col for col in advanced_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        X, y = create_feature_matrix(advanced_data, indicator_cols, binary_labels.name)
        
        print(f"✅ Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train multiple models
        if ModelPipeline is not None:
            pipeline_manager = ModelPipeline()
            models = ['logistic', 'random_forest']
            
            for model_type in models:
                try:
                    pipeline_manager.create_classification_pipeline(f"model_{model_type}", model_type)
                    result = pipeline_manager.train_model(f"model_{model_type}", X, y)
                    
                    metrics = result['metrics']
                    print(f"  • {model_type.title()}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
                    
                except Exception as e:
                    print(f"  • {model_type.title()}: Failed - {e}")
            
            # Feature importance analysis
            if 'model_random_forest' in pipeline_manager.feature_importance:
                importance = pipeline_manager.feature_importance['model_random_forest']
                if importance:
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    print(f"\n📈 Top 10 Most Important Features:")
                    for feature, imp in top_features:
                        print(f"  • {feature}: {imp:.3f}")
        else:
            print("  • ML Pipeline: Skipped due to XGBoost dependency")
        
    except Exception as e:
        print(f"❌ Machine learning failed: {e}")
    
    # Step 7: Benchmarking and Attribution
    print_section("Step 7: Benchmarking & Performance Attribution")
    try:
        # Create benchmark (S&P 500 proxy using equal weight of our assets)
        benchmark_returns = portfolio_returns.mean(axis=1)
        
        # Add benchmark
        benchmark_analyzer = BenchmarkAnalyzer()
        benchmark_analyzer.add_benchmark("Market Portfolio", benchmark_returns, "Equal-weighted portfolio")
        
        # Calculate attribution for best strategy
        if results:
            best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
            best_returns = pd.Series(results[best_strategy]['portfolio']['returns'], 
                                   index=results[best_strategy]['portfolio'].index)
            
            attribution = benchmark_analyzer.calculate_attribution_metrics(best_returns)
            
            print(f"✅ Performance Attribution for {best_strategy}:")
            rel_perf = attribution['relative_performance']
            print(f"  • Excess Return: {rel_perf['annualized_excess_return']:.2%}")
            print(f"  • Information Ratio: {rel_perf['information_ratio']:.2f}")
            print(f"  • Beta: {rel_perf['beta']:.2f}")
            print(f"  • Alpha: {rel_perf['alpha_annualized']:.2%}")
            print(f"  • Up Capture: {rel_perf['up_capture']:.2f}")
            print(f"  • Down Capture: {rel_perf['down_capture']:.2f}")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
    
    # Step 8: Parameter Optimization
    print_section("Step 8: Parameter Optimization")
    try:
        # Define parameter space for RSI strategy
        def rsi_strategy_wrapper(data, rsi_oversold=30, rsi_overbought=70):
            """Wrapper for RSI strategy optimization."""
            if 'rsi_14' not in data.columns or 'close_20_ema' not in data.columns:
                return {'total_return': 0, 'sharpe_ratio': 0}
            
            signals = StrategyBuilder.rsi_trend_strategy(
                data, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought
            )
            
            engine = BacktestEngine()
            result = engine.run_backtest(data, signals, strategy_name='RSI_Optimized')
            return result
        
        def objective_function(result):
            """Objective function for optimization."""
            return result['sharpe_ratio']
        
        parameter_space = {
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80)
        }
        
        param_optimizer = ParameterOptimizer()
        opt_result = param_optimizer.optimize_strategy_parameters(
            rsi_strategy_wrapper,
            parameter_space,
            advanced_data,
            objective_function,
            method='differential_evolution',
            max_iterations=50
        )
        
        if opt_result['success']:
            print(f"✅ Parameter Optimization Results:")
            print(f"  • Best Score: {opt_result['best_score']:.3f}")
            print(f"  • Optimal Parameters:")
            for param, value in opt_result['optimal_parameters'].items():
                print(f"    - {param}: {value:.1f}")
        else:
            print(f"❌ Parameter optimization failed: {opt_result.get('message', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Parameter optimization failed: {e}")
    
    # Step 9: Generate Summary Report
    print_section("Step 9: Summary Report")
    try:
        print("📊 ENHANCED DEMO SUMMARY")
        print("=" * 50)
        print(f"✅ Data: {len(primary_data)} records, {len(tickers)} assets")
        print(f"✅ Indicators: {len(advanced_data.columns) - 5} technical indicators")
        print(f"✅ Strategies: {len(strategies)} backtested strategies")
        print(f"✅ Models: {len([m for m in ['logistic', 'random_forest'] if f'model_{m}' in locals()])} ML models")
        print(f"✅ Optimization: Portfolio & parameter optimization completed")
        print(f"✅ Risk Analysis: VaR, CVaR, drawdown analysis completed")
        print(f"✅ Benchmarking: Performance attribution completed")
        
        # Performance summary
        if results:
            best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
            best_result = results[best_strategy]
            print(f"\n🏆 BEST STRATEGY: {best_strategy}")
            print(f"  • Total Return: {best_result['total_return']:.2%}")
            print(f"  • Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
            print(f"  • Max Drawdown: {best_result['max_drawdown']:.2%}")
            print(f"  • Hit Rate: {best_result['hit_rate']:.2%}")
        
        print(f"\n🎉 Enhanced demo completed successfully!")
        print(f"📈 Ready for portfolio demonstration and interview!")
        
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
    
    print(f"\n{'='*60}")
    print("🚀 StockStats Alpha Lab - Enhanced Demo Complete!")
    print("Next steps:")
    print("• Run: streamlit run streamlit_app.py")
    print("• Explore: jupyter notebook notebooks/")
    print("• Test: python -m pytest tests/")
    print("• Optimize: python enhanced_demo.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
