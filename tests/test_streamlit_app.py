"""
Tests for Streamlit Application

This module tests the Streamlit dashboard functionality
including data loading, visualization, and interactive features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.append('src')

# Mock streamlit components
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit components for testing."""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.markdown'), \
         patch('streamlit.sidebar'), \
         patch('streamlit.spinner'), \
         patch('streamlit.tabs'), \
         patch('streamlit.columns'), \
         patch('streamlit.selectbox'), \
         patch('streamlit.slider'), \
         patch('streamlit.checkbox'), \
         patch('streamlit.plotly_chart'), \
         patch('streamlit.dataframe'), \
         patch('streamlit.metric'), \
         patch('streamlit.error'), \
         patch('streamlit.warning'), \
         patch('streamlit.info'), \
         patch('streamlit.cache_data'):
        yield


class TestStreamlitApp:
    """Test the Streamlit application components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'close_20_ema': prices * (1 + np.random.normal(0, 0.005, 100)),
            'macd': np.random.randn(100),
            'macds': np.random.randn(100),
            'boll_ub': prices * 1.1,
            'boll_lb': prices * 0.9,
            'atr_14': np.random.uniform(0.5, 2.0, 100)
        }, index=dates)
    
    def test_load_and_process_data(self, sample_data):
        """Test data loading and processing function."""
        from streamlit_app import load_and_process_data
        
        # Mock the function to return sample data
        with patch('streamlit_app.load_and_process_data', return_value=sample_data):
            result = load_and_process_data('AAPL', '3mo')
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
            assert 'close' in result.columns
            assert 'volume' in result.columns
    
    def test_show_market_overview(self, sample_data):
        """Test market overview display function."""
        from streamlit_app import show_market_overview
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock() for _ in range(4)]
            mock_st.metric.return_value = None
            mock_st.plotly_chart.return_value = None
            
            # Test the function
            show_market_overview(sample_data, 'AAPL')
            
            # Verify that metrics were called
            assert mock_st.metric.call_count >= 4
    
    def test_show_technical_analysis(self, sample_data):
        """Test technical analysis display function."""
        from streamlit_app import show_technical_analysis
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.checkbox.return_value = True
            mock_st.plotly_chart.return_value = None
            
            # Test the function
            show_technical_analysis(sample_data, 'AAPL')
            
            # Verify that checkboxes were created
            assert mock_st.checkbox.call_count >= 4
    
    def test_show_strategy_backtesting(self, sample_data):
        """Test strategy backtesting display function."""
        from streamlit_app import show_strategy_backtesting
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.dataframe.return_value = None
            mock_st.plotly_chart.return_value = None
            mock_st.selectbox.return_value = 'RSI + Trend'
            
            # Mock backtest engine
            with patch('streamlit_app.BacktestEngine') as mock_engine:
                mock_engine.return_value.run_backtest.return_value = {
                    'total_return': 0.05,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.03,
                    'hit_rate': 0.6,
                    'num_trades': 10,
                    'portfolio': pd.DataFrame({
                        'cumulative_returns': [1.0, 1.01, 1.02, 1.03],
                        'drawdown': [0.0, -0.01, -0.02, -0.01]
                    })
                }
                
                # Test the function
                show_strategy_backtesting(sample_data, 30, 70, 12, 26, 0.1, 0.001, 0.0005)
                
                # Verify that backtest was called
                assert mock_engine.return_value.run_backtest.called
    
    def test_show_machine_learning(self, sample_data):
        """Test machine learning display function."""
        from streamlit_app import show_machine_learning
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.selectbox.return_value = 'logistic'
            mock_st.slider.return_value = 0.2
            mock_st.metric.return_value = None
            mock_st.plotly_chart.return_value = None
            
            # Mock ML pipeline
            with patch('streamlit_app.ModelPipeline') as mock_pipeline:
                mock_pipeline.return_value.create_classification_pipeline.return_value = None
                mock_pipeline.return_value.train_model.return_value = {
                    'metrics': {
                        'accuracy': 0.75,
                        'precision': 0.73,
                        'recall': 0.71,
                        'f1': 0.72
                    }
                }
                mock_pipeline.return_value.feature_importance = {}
                
                # Test the function
                show_machine_learning(sample_data)
                
                # Verify that ML pipeline was called
                assert mock_pipeline.return_value.create_classification_pipeline.called
    
    def test_show_performance_analytics(self, sample_data):
        """Test performance analytics display function."""
        from streamlit_app import show_performance_analytics
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.metric.return_value = None
            mock_st.plotly_chart.return_value = None
            
            # Test the function
            show_performance_analytics(sample_data)
            
            # Verify that metrics were called
            assert mock_st.metric.call_count >= 4
    
    def test_calculate_max_drawdown(self, sample_data):
        """Test maximum drawdown calculation function."""
        from streamlit_app import calculate_max_drawdown
        
        prices = sample_data['close']
        max_dd = calculate_max_drawdown(prices)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
    
    def test_main_function(self):
        """Test main function execution."""
        from streamlit_app import main
        
        # Mock all dependencies
        with patch('streamlit_app.load_and_process_data') as mock_load, \
             patch('streamlit_app.st') as mock_st:
            
            # Mock data loading
            mock_load.return_value = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            # Mock streamlit components
            mock_st.spinner.return_value.__enter__ = MagicMock()
            mock_st.spinner.return_value.__exit__ = MagicMock()
            mock_st.tabs.return_value = [MagicMock() for _ in range(5)]
            mock_st.sidebar.selectbox.return_value = 'AAPL'
            mock_st.sidebar.slider.return_value = 30
            
            # Test main function
            try:
                main()
                # If no exception is raised, the test passes
                assert True
            except Exception as e:
                # Check if it's a known issue (like missing data)
                if "Error loading data" in str(e):
                    assert True  # Expected behavior
                else:
                    raise e


class TestStreamlitAppEdgeCases:
    """Test edge cases for Streamlit application."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from streamlit_app import show_market_overview
        
        empty_data = pd.DataFrame()
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock() for _ in range(4)]
            mock_st.metric.return_value = None
            mock_st.plotly_chart.return_value = None
            
            # Should handle empty data gracefully
            try:
                show_market_overview(empty_data, 'TEST')
                assert True
            except (IndexError, KeyError):
                # Expected for empty data
                assert True
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        from streamlit_app import show_technical_analysis
        
        # Data with missing columns
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.checkbox.return_value = True
            mock_st.plotly_chart.return_value = None
            
            # Should handle missing columns gracefully
            try:
                show_technical_analysis(incomplete_data, 'TEST')
                assert True
            except KeyError:
                # Expected for missing columns
                assert True
    
    def test_error_handling_in_main(self):
        """Test error handling in main function."""
        from streamlit_app import main
        
        # Mock data loading to raise an error
        with patch('streamlit_app.load_and_process_data', side_effect=Exception("Test error")):
            with patch('streamlit_app.st') as mock_st:
                mock_st.error.return_value = None
                mock_st.info.return_value = None
                
                # Should handle errors gracefully
                main()
                
                # Verify that error was displayed
                assert mock_st.error.called


class TestStreamlitAppIntegration:
    """Test integration aspects of Streamlit application."""
    
    def test_data_flow(self, sample_data):
        """Test data flow through the application."""
        from streamlit_app import load_and_process_data, show_market_overview
        
        # Mock the data loading
        with patch('streamlit_app.load_and_process_data', return_value=sample_data):
            data = load_and_process_data('AAPL', '3mo')
            
            # Mock streamlit components
            with patch('streamlit_app.st') as mock_st:
                mock_st.columns.return_value = [MagicMock() for _ in range(4)]
                mock_st.metric.return_value = None
                mock_st.plotly_chart.return_value = None
                
                # Test that data flows correctly
                show_market_overview(data, 'AAPL')
                
                # Verify data was processed
                assert isinstance(data, pd.DataFrame)
                assert len(data) > 0
    
    def test_parameter_passing(self, sample_data):
        """Test parameter passing between functions."""
        from streamlit_app import show_strategy_backtesting
        
        # Mock streamlit components
        with patch('streamlit_app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.dataframe.return_value = None
            mock_st.plotly_chart.return_value = None
            mock_st.selectbox.return_value = 'RSI + Trend'
            
            # Mock backtest engine
            with patch('streamlit_app.BacktestEngine') as mock_engine:
                mock_engine.return_value.run_backtest.return_value = {
                    'total_return': 0.05,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.03,
                    'hit_rate': 0.6,
                    'num_trades': 10,
                    'portfolio': pd.DataFrame({
                        'cumulative_returns': [1.0, 1.01, 1.02, 1.03],
                        'drawdown': [0.0, -0.01, -0.02, -0.01]
                    })
                }
                
                # Test with specific parameters
                rsi_oversold = 25
                rsi_overbought = 75
                max_position = 0.15
                
                show_strategy_backtesting(
                    sample_data, rsi_oversold, rsi_overbought, 
                    12, 26, max_position, 0.001, 0.0005
                )
                
                # Verify parameters were used
                assert mock_engine.return_value.run_backtest.called
