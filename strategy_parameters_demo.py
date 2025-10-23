#!/usr/bin/env python3
"""
Strategy Parameters & Risk Management Demo

This script demonstrates how to use different strategy parameters
and risk management settings in the StockStats Alpha Lab.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from data import DataLoader
from indicators import add_basic_indicators
from backtest import BacktestEngine, StrategyBuilder

def demonstrate_strategy_parameters():
    """Demonstrate different strategy parameter settings."""
    
    print("🎯 Strategy Parameters & Risk Management Demo")
    print("=" * 50)
    
    # Load sample data
    print("📊 Loading sample data...")
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="1y")
    df = add_basic_indicators(df)
    
    print(f"✅ Loaded {len(df)} days of data for AAPL")
    print()
    
    # Define different parameter sets
    parameter_sets = {
        "Conservative": {
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "max_position": 0.08,  # 8%
            "commission": 0.001,   # 0.1%
            "slippage": 0.0005,     # 0.05%
            "description": "Lower risk, smaller positions, higher RSI thresholds"
        },
        "Balanced": {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "max_position": 0.12,  # 12%
            "commission": 0.001,    # 0.1%
            "slippage": 0.0005,    # 0.05%
            "description": "Default settings, moderate risk and reward"
        },
        "Aggressive": {
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "max_position": 0.20,  # 20%
            "commission": 0.001,   # 0.1%
            "slippage": 0.0005,    # 0.05%
            "description": "Higher risk, larger positions, lower RSI thresholds"
        }
    }
    
    # Test each parameter set
    results = {}
    
    for strategy_name, params in parameter_sets.items():
        print(f"🔧 Testing {strategy_name} Strategy:")
        print(f"   {params['description']}")
        print(f"   RSI Oversold: {params['rsi_oversold']}, Overbought: {params['rsi_overbought']}")
        print(f"   Max Position: {params['max_position']:.1%}")
        print()
        
        try:
            # Create RSI strategy with these parameters
            signals = StrategyBuilder.rsi_trend_strategy(
                df, 
                rsi_oversold=params['rsi_oversold'],
                rsi_overbought=params['rsi_overbought']
            )
            
            # Run backtest
            engine = BacktestEngine(
                max_position_size=params['max_position'],
                commission=params['commission'],
                slippage=params['slippage']
            )
            
            result = engine.run_backtest(df, signals, strategy_name=strategy_name)
            results[strategy_name] = result
            
            # Display key metrics
            print(f"   📈 Total Return: {result['total_return']:.2%}")
            print(f"   📊 Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"   ⚠️  Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"   🎯 Hit Rate: {result['hit_rate']:.1%}")
            print(f"   💰 Number of Trades: {result['num_trades']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            print()
    
    # Compare results
    print("📊 Strategy Comparison Summary:")
    print("-" * 50)
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Strategy': name,
            'Total Return': f"{result['total_return']:.2%}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{result['max_drawdown']:.2%}",
            'Hit Rate': f"{result['hit_rate']:.1%}",
            'Trades': result['num_trades']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # Risk analysis
    print("⚠️ Risk Analysis:")
    print("-" * 20)
    
    best_return = max(results.values(), key=lambda x: x['total_return'])
    best_sharpe = max(results.values(), key=lambda x: x['sharpe_ratio'])
    lowest_drawdown = min(results.values(), key=lambda x: x['max_drawdown'])
    
    print(f"🏆 Highest Return: {best_return['total_return']:.2%}")
    print(f"🎯 Best Risk-Adjusted Return (Sharpe): {best_sharpe['sharpe_ratio']:.2f}")
    print(f"🛡️ Lowest Drawdown: {lowest_drawdown['max_drawdown']:.2%}")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print("-" * 20)
    
    if best_sharpe['sharpe_ratio'] > 1.0:
        print("✅ Good risk-adjusted returns achieved!")
    else:
        print("⚠️ Consider adjusting parameters for better risk-adjusted returns")
    
    if lowest_drawdown['max_drawdown'] < 0.1:
        print("✅ Drawdowns are manageable")
    else:
        print("⚠️ Consider more conservative position sizing")
    
    print()
    print("🎯 Key Takeaways:")
    print("1. Conservative strategies have lower drawdowns but may miss opportunities")
    print("2. Aggressive strategies can have higher returns but also higher risks")
    print("3. Balanced strategies often provide the best risk-adjusted returns")
    print("4. Always test parameters thoroughly before using real money")
    print("5. Consider your risk tolerance when choosing parameters")

def demonstrate_risk_management():
    """Demonstrate risk management concepts."""
    
    print("\n" + "=" * 50)
    print("🛡️ Risk Management Demonstration")
    print("=" * 50)
    
    # Load data
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="2y")
    df = add_basic_indicators(df)
    
    # Calculate returns
    returns = df['close'].pct_change().dropna()
    
    print("📊 Risk Metrics for AAPL (2 years):")
    print()
    
    # VaR calculations
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    print(f"⚠️ Value at Risk (VaR):")
    print(f"   95% VaR: {var_95:.2%} (5% chance of losing more than this)")
    print(f"   99% VaR: {var_99:.2%} (1% chance of losing more than this)")
    print()
    
    # CVaR calculations
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    print(f"📉 Conditional Value at Risk (CVaR):")
    print(f"   95% CVaR: {cvar_95:.2%} (average loss when losing more than 95% VaR)")
    print(f"   99% CVaR: {cvar_99:.2%} (average loss when losing more than 99% VaR)")
    print()
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)  # Annualized
    print(f"📊 Annualized Volatility: {volatility:.2%}")
    print()
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"📉 Maximum Drawdown: {max_drawdown:.2%}")
    print()
    
    # Risk management recommendations
    print("💡 Risk Management Recommendations:")
    print("-" * 40)
    
    if volatility > 0.3:
        print("⚠️ High volatility detected - consider smaller position sizes")
    elif volatility < 0.15:
        print("✅ Low volatility - can consider larger position sizes")
    else:
        print("✅ Moderate volatility - standard position sizing appropriate")
    
    if max_drawdown < -0.2:
        print("⚠️ Large drawdowns possible - implement strict stop losses")
    elif max_drawdown > -0.1:
        print("✅ Manageable drawdowns - standard risk management sufficient")
    else:
        print("⚠️ Moderate drawdowns - consider conservative position sizing")
    
    print()
    print("🎯 Risk Management Best Practices:")
    print("1. Never risk more than 2% of your portfolio on a single trade")
    print("2. Use stop losses to limit downside risk")
    print("3. Diversify across different stocks and sectors")
    print("4. Regularly review and adjust your risk parameters")
    print("5. Consider your emotional tolerance for losses")

if __name__ == "__main__":
    try:
        demonstrate_strategy_parameters()
        demonstrate_risk_management()
        
        print("\n" + "=" * 50)
        print("🎉 Demo Complete!")
        print("=" * 50)
        print("📚 For more detailed explanations, see:")
        print("   • STRATEGY_PARAMETERS_GUIDE.md")
        print("   • DASHBOARD_USER_GUIDE.md")
        print("   • QUICK_START_GUIDE.md")
        print()
        print("🌐 Try these settings in the dashboard at: http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        print("Make sure the dashboard is running and all dependencies are installed.")
