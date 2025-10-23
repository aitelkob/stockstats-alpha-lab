"""
Tests for backtest module.

This module tests the backtesting functionality to ensure proper
strategy execution and performance calculation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest import (
    BacktestEngine,
    StrategyBuilder,
    run_strategy_comparison,
    stress_test_strategy
)


class TestBacktest:
    """Test class for backtesting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create realistic price data
        np.random.seed(42)
        price = 100
        prices = []
        
        for _ in range(100):
            price += np.random.normal(0, 0.02) * price
            prices.append(price)
        
        prices = np.array(prices)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'close_200_sma': prices * (1 + np.random.normal(0, 0.05, 100)),
            'macd': np.random.randn(100),
            'macds': np.random.randn(100),
            'atr_14': np.random.uniform(0.5, 2.0, 100),
            'boll_ub': prices * 1.1,
            'boll_lb': prices * 0.9
        }, index=dates)
        
        # Ensure high >= low, high >= open, high >= close
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create random signals
        np.random.seed(42)
        signals = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)
        
        return signals
    
    def test_backtest_engine_initialization(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            max_position_size=0.1
        )
        
        assert engine.initial_capital == 100000
        assert engine.commission == 0.001
        assert engine.slippage == 0.0005
        assert engine.max_position_size == 0.1
    
    def test_run_backtest(self, sample_data, sample_signals):
        """Test basic backtest execution."""
        engine = BacktestEngine()
        results = engine.run_backtest(sample_data, sample_signals, strategy_name="test_strategy")
        
        # Check that results contain expected keys
        expected_keys = [
            'strategy_name', 'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'hit_rate',
            'calmar_ratio', 'turnover', 'var_95', 'es_95', 'num_trades',
            'final_capital', 'portfolio'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check that portfolio DataFrame has expected columns
        portfolio = results['portfolio']
        expected_portfolio_cols = [
            'capital', 'position', 'shares', 'cash', 'returns',
            'cumulative_returns', 'drawdown', 'max_drawdown'
        ]
        
        for col in expected_portfolio_cols:
            assert col in portfolio.columns, f"Missing portfolio column: {col}"
        
        # Check that final capital is reasonable
        assert results['final_capital'] > 0
        assert results['final_capital'] != engine.initial_capital  # Should have changed
    
    def test_rsi_trend_strategy(self, sample_data):
        """Test RSI + Trend strategy."""
        signals = StrategyBuilder.rsi_trend_strategy(sample_data)
        
        # Check that signals are valid
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(signals) == len(sample_data)
        
        # Check that we have some non-zero signals
        assert (signals != 0).any()
    
    def test_macd_crossover_strategy(self, sample_data):
        """Test MACD crossover strategy."""
        signals = StrategyBuilder.macd_crossover_strategy(sample_data)
        
        # Check that signals are valid
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(signals) == len(sample_data)
    
    def test_bollinger_bands_strategy(self, sample_data):
        """Test Bollinger Bands strategy."""
        signals = StrategyBuilder.bollinger_bands_strategy(sample_data)
        
        # Check that signals are valid
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(signals) == len(sample_data)
    
    def test_volatility_sizing_strategy(self, sample_data):
        """Test volatility sizing strategy."""
        base_signals = pd.Series([1, -1, 0, 1, -1] * 20, index=sample_data.index[:100])
        sized_signals = StrategyBuilder.volatility_sizing_strategy(sample_data, base_signals)
        
        # Check that sized signals are different from base signals
        assert not sized_signals.equals(base_signals)
        
        # Check that sized signals have reasonable values
        assert sized_signals.abs().max() <= 1.0  # Should be normalized
    
    def test_transaction_costs_impact(self, sample_data, sample_signals):
        """Test that transaction costs affect performance."""
        # Run backtest with high transaction costs
        engine_high_cost = BacktestEngine(commission=0.01, slippage=0.005)
        results_high = engine_high_cost.run_backtest(sample_data, sample_signals, strategy_name="high_cost")
        
        # Run backtest with low transaction costs
        engine_low_cost = BacktestEngine(commission=0.0001, slippage=0.0001)
        results_low = engine_low_cost.run_backtest(sample_data, sample_signals, strategy_name="low_cost")
        
        # High cost strategy should have lower returns
        assert results_high['total_return'] <= results_low['total_return']
    
    def test_position_sizing_limits(self, sample_data):
        """Test that position sizing respects limits."""
        # Create signals that would exceed position limits
        extreme_signals = pd.Series([2, -2, 2, -2] * 25, index=sample_data.index[:100])
        
        engine = BacktestEngine(max_position_size=0.1)
        results = engine.run_backtest(sample_data, extreme_signals, strategy_name="extreme")
        
        # Check that positions don't exceed limits
        portfolio = results['portfolio']
        assert portfolio['position'].abs().max() <= engine.max_position_size * 1.1  # Allow small tolerance
    
    def test_strategy_comparison(self, sample_data):
        """Test strategy comparison functionality."""
        # Create different strategies
        strategies = {
            'rsi_trend': StrategyBuilder.rsi_trend_strategy(sample_data),
            'macd_crossover': StrategyBuilder.macd_crossover_strategy(sample_data),
            'bollinger_bands': StrategyBuilder.bollinger_bands_strategy(sample_data)
        }
        
        # Run comparison
        comparison_df = run_strategy_comparison(sample_data, strategies)
        
        # Check that we get results for all strategies
        assert len(comparison_df) == len(strategies)
        
        # Check that comparison has expected columns
        expected_cols = ['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown']
        for col in expected_cols:
            assert col in comparison_df.columns
    
    def test_stress_testing(self, sample_data, sample_signals):
        """Test stress testing functionality."""
        # Define stress periods
        stress_periods = [
            ('crisis_1', '2023-01-01', '2023-01-31'),
            ('crisis_2', '2023-02-01', '2023-02-28')
        ]
        
        # Run stress test
        stress_results = stress_test_strategy(sample_data, sample_signals, stress_periods)
        
        # Check that we get results for stress periods
        assert len(stress_results) == len(stress_periods)
        
        # Check that each stress result has expected structure
        for period_name, result in stress_results.items():
            assert 'total_return' in result
            assert 'sharpe_ratio' in result
            assert 'max_drawdown' in result
    
    def test_performance_metrics_calculation(self, sample_data, sample_signals):
        """Test that performance metrics are calculated correctly."""
        engine = BacktestEngine()
        results = engine.run_backtest(sample_data, sample_signals, strategy_name="test")
        
        # Check Sharpe ratio calculation
        portfolio = results['portfolio']
        returns = portfolio['returns'].dropna()
        
        if returns.std() > 0:
            expected_sharpe = returns.mean() / returns.std() * np.sqrt(252)
            assert abs(results['sharpe_ratio'] - expected_sharpe) < 0.01
        
        # Check that max drawdown is non-negative
        assert results['max_drawdown'] >= 0
        
        # Check that hit rate is between 0 and 1
        assert 0 <= results['hit_rate'] <= 1
        
        # Check that turnover is non-negative
        assert results['turnover'] >= 0
    
    def test_portfolio_tracking(self, sample_data, sample_signals):
        """Test that portfolio values are tracked correctly."""
        engine = BacktestEngine()
        results = engine.run_backtest(sample_data, sample_signals, strategy_name="test")
        
        portfolio = results['portfolio']
        
        # Check that capital is always positive
        assert (portfolio['capital'] > 0).all()
        
        # Check that cash + shares * price = capital (approximately)
        portfolio_value = portfolio['cash'] + portfolio['shares'] * sample_data['close']
        assert np.allclose(portfolio['capital'], portfolio_value, rtol=1e-10)
        
        # Check that cumulative returns start at 0
        assert portfolio['cumulative_returns'].iloc[0] == 0
        
        # Check that drawdown is non-negative
        assert (portfolio['drawdown'] >= 0).all()


class TestBacktestEdgeCases:
    """Test edge cases for backtesting."""
    
    def test_empty_signals(self):
        """Test handling of empty signals."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })
        
        empty_signals = pd.Series([], dtype=int)
        
        engine = BacktestEngine()
        
        with pytest.raises(Exception):
            engine.run_backtest(data, empty_signals)
    
    def test_all_zero_signals(self):
        """Test handling of all-zero signals."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })
        
        zero_signals = pd.Series([0, 0, 0])
        
        engine = BacktestEngine()
        results = engine.run_backtest(data, zero_signals, strategy_name="zero")
        
        # Should have no trades
        assert results['num_trades'] == 0
        assert results['total_return'] == 0
    
    def test_single_row_data(self):
        """Test handling of single-row data."""
        data = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000000]
        })
        
        signals = pd.Series([1])
        
        engine = BacktestEngine()
        results = engine.run_backtest(data, signals, strategy_name="single")
        
        # Should complete without error
        assert 'total_return' in results
    
    def test_mismatched_indices(self):
        """Test handling of mismatched data and signal indices."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        signals = pd.Series([1, -1, 1], index=pd.date_range('2023-01-02', periods=3))
        
        engine = BacktestEngine()
        
        # Should handle mismatched indices gracefully
        results = engine.run_backtest(data, signals, strategy_name="mismatched")
        assert 'total_return' in results


if __name__ == "__main__":
    pytest.main([__file__])
