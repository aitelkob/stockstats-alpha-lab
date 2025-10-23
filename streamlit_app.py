"""
Interactive Streamlit Dashboard for StockStats Alpha Lab

A comprehensive web application for quantitative finance analysis,
strategy backtesting, and real-time market insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append('src')

from data import DataLoader
from indicators import add_basic_indicators, add_comprehensive_indicators, IndicatorEngine
from labeling import LabelingEngine, calculate_information_coefficient
from backtest import BacktestEngine, StrategyBuilder, run_strategy_comparison
from models import ModelPipeline
from plots import Plotter

# Page configuration
st.set_page_config(
    page_title="StockStats Alpha Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .strategy-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 StockStats Alpha Lab</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Quantitative Finance Research Platform")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Data selection
    st.sidebar.header("📊 Data Selection")
    ticker = st.sidebar.selectbox(
        "Select Ticker",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
        index=0
    )
    
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )
    
    # Strategy parameters
    st.sidebar.header("⚙️ Strategy Parameters")
    rsi_oversold = st.sidebar.slider("RSI Oversold", 20, 40, 30)
    rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 80, 70)
    macd_fast = st.sidebar.slider("MACD Fast Period", 8, 16, 12)
    macd_slow = st.sidebar.slider("MACD Slow Period", 20, 30, 26)
    
    # Risk parameters
    st.sidebar.header("🛡️ Risk Management")
    max_position = st.sidebar.slider("Max Position Size", 0.05, 0.5, 0.1)
    commission = st.sidebar.slider("Commission (%)", 0.0, 0.5, 0.1) / 100
    slippage = st.sidebar.slider("Slippage (%)", 0.0, 0.2, 0.05) / 100
    
    # Load data
    @st.cache_data
    def load_and_process_data(ticker, period):
        """Load and process market data."""
        loader = DataLoader()
        df = loader.load_single_ticker(ticker, period=period)
        df = add_comprehensive_indicators(df)
        return df
    
    try:
        with st.spinner(f"Loading {ticker} data for {period}..."):
            df = load_and_process_data(ticker, period)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Market Overview", 
            "🔧 Technical Analysis", 
            "💰 Strategy Backtesting", 
            "🤖 Machine Learning", 
            "📈 Performance Analytics"
        ])
        
        with tab1:
            show_market_overview(df, ticker)
        
        with tab2:
            show_technical_analysis(df, ticker)
        
        with tab3:
            show_strategy_backtesting(df, rsi_oversold, rsi_overbought, macd_fast, macd_slow, max_position, commission, slippage)
        
        with tab4:
            show_machine_learning(df)
        
        with tab5:
            show_performance_analytics(df)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check your internet connection and try again.")

def show_market_overview(df, ticker):
    """Display market overview and basic statistics."""
    st.header(f"📊 {ticker} Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col3:
        volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Annualized Volatility", f"{volatility:.2f}%")
    
    with col4:
        max_dd = calculate_max_drawdown(df['close'])
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages
    if 'close_20_sma' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close_20_sma'],
            mode='lines',
            name='20 SMA',
            line=dict(color='orange', width=1, dash='dash')
        ))
    
    if 'close_50_sma' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close_50_sma'],
            mode='lines',
            name='50 SMA',
            line=dict(color='red', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{ticker} Price Chart with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig_vol.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)

def show_technical_analysis(df, ticker):
    """Display technical analysis charts and indicators."""
    st.header(f"🔧 Technical Analysis - {ticker}")
    
    # Indicator selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Select Indicators")
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_bollinger = st.checkbox("Bollinger Bands", value=True)
        show_atr = st.checkbox("ATR", value=False)
    
    with col2:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD', 'ATR'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ), row=1, col=1)
        
        if show_bollinger and 'boll_ub' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['boll_ub'],
                mode='lines',
                name='Upper BB',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['boll_lb'],
                mode='lines',
                name='Lower BB',
                line=dict(color='red', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # RSI
        if show_rsi and 'rsi_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['rsi_14'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ), row=2, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        if show_macd and 'macd' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ), row=3, col=1)
            
            if 'macds' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['macds'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red', width=2)
                ), row=3, col=1)
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        
        # ATR
        if show_atr and 'atr_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['atr_14'],
                mode='lines',
                name='ATR',
                line=dict(color='orange', width=2)
            ), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def show_strategy_backtesting(df, rsi_oversold, rsi_overbought, macd_fast, macd_slow, max_position, commission, slippage):
    """Display strategy backtesting results."""
    st.header("💰 Strategy Backtesting")
    
    # Create strategies
    strategies = {}
    
    # RSI Strategy
    if 'rsi_14' in df.columns and 'close_20_ema' in df.columns:
        strategies['RSI + Trend'] = StrategyBuilder.rsi_trend_strategy(
            df, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought
        )
    
    # MACD Strategy
    if 'macd' in df.columns and 'macds' in df.columns:
        strategies['MACD Crossover'] = StrategyBuilder.macd_crossover_strategy(df)
    
    # Bollinger Bands Strategy
    if all(col in df.columns for col in ['boll_ub', 'boll_lb']):
        strategies['Bollinger Bands'] = StrategyBuilder.bollinger_bands_strategy(df)
    
    if not strategies:
        st.warning("No strategies can be created with the available data.")
        return
    
    # Run backtests
    engine = BacktestEngine(
        max_position_size=max_position,
        commission=commission,
        slippage=slippage
    )
    
    results = {}
    for name, signals in strategies.items():
        try:
            result = engine.run_backtest(df, signals, strategy_name=name)
            results[name] = result
        except Exception as e:
            st.error(f"Error running {name}: {str(e)}")
    
    if not results:
        st.error("No strategies could be backtested successfully.")
        return
    
    # Display results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Strategy Performance")
        
        # Create comparison table
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
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        st.subheader("Equity Curves")
        
        # Plot equity curves
        fig = go.Figure()
        
        for name, result in results.items():
            portfolio = result['portfolio']
            fig.add_trace(go.Scatter(
                x=portfolio.index,
                y=portfolio['cumulative_returns'] * 100,
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Strategy Equity Curves",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.subheader("Detailed Analysis")
    
    selected_strategy = st.selectbox("Select Strategy for Detailed Analysis", list(results.keys()))
    
    if selected_strategy:
        result = results[selected_strategy]
        portfolio = result['portfolio']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{result['total_return']:.2%}")
            st.metric("Annualized Return", f"{result['annualized_return']:.2%}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{result['sortino_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
            st.metric("Hit Rate", f"{result['hit_rate']:.2%}")
        
        # Drawdown chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=portfolio.index,
            y=portfolio['drawdown'] * 100,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=2)
        ))
        
        fig_dd.update_layout(
            title=f"{selected_strategy} - Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)

def show_machine_learning(df):
    """Display machine learning analysis."""
    st.header("🤖 Machine Learning Analysis")
    
    # Create labels
    labeler = LabelingEngine()
    forward_returns = labeler.forward_return_label(df, horizon=5)
    binary_labels = labeler.binary_classification_label(forward_returns)
    
    # Feature selection
    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    if not indicator_cols:
        st.warning("No technical indicators available for ML analysis.")
        return
    
    # Create feature matrix
    from labeling import create_feature_matrix
    X, y = create_feature_matrix(df, indicator_cols, binary_labels.name)
    
    if X.empty or y.empty:
        st.warning("Insufficient data for ML analysis.")
        return
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        model_type = st.selectbox("Select Model", ["logistic", "random_forest"])
        validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)
    
    with col2:
        # Train model
        pipeline_manager = ModelPipeline()
        pipeline_manager.create_classification_pipeline("ml_model", model_type)
        
        try:
            result = pipeline_manager.train_model("ml_model", X, y, validation_split)
            
            # Display metrics
            metrics = result['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            
            with col4:
                st.metric("F1 Score", f"{metrics['f1']:.3f}")
            
            # Feature importance
            if 'ml_model' in pipeline_manager.feature_importance:
                importance = pipeline_manager.feature_importance['ml_model']
                
                if importance:
                    importance_df = pd.DataFrame(
                        list(importance.items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        importance_df.tail(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Feature Importance"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

def show_performance_analytics(df):
    """Display performance analytics and risk metrics."""
    st.header("📈 Performance Analytics")
    
    # Calculate returns
    returns = df['close'].pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")
        
        # VaR calculation
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        st.metric("VaR (95%)", f"{var_95:.2%}")
        st.metric("VaR (99%)", f"{var_99:.2%}")
        
        # CVaR
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        st.metric("CVaR (95%)", f"{cvar_95:.2%}")
        st.metric("CVaR (99%)", f"{cvar_99:.2%}")
    
    with col2:
        st.subheader("Performance Metrics")
        
        # Basic metrics
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        annualized_return = total_return * (252 / len(df))
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        st.metric("Total Return", f"{total_return:.2f}%")
        st.metric("Annualized Return", f"{annualized_return:.2f}%")
        st.metric("Volatility", f"{volatility:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Returns distribution
    fig = px.histogram(
        returns * 100,
        nbins=50,
        title="Returns Distribution",
        labels={'x': 'Daily Returns (%)', 'y': 'Frequency'}
    )
    fig.add_vline(x=var_95*100, line_dash="dash", line_color="red", annotation_text="VaR 95%")
    fig.add_vline(x=var_99*100, line_dash="dash", line_color="darkred", annotation_text="VaR 99%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling metrics
    window = 30
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    rolling_sharpe = (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))
    
    fig_rolling = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Volatility', 'Rolling Sharpe Ratio'),
        vertical_spacing=0.1
    )
    
    fig_rolling.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        mode='lines',
        name='Volatility',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig_rolling.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        mode='lines',
        name='Sharpe Ratio',
        line=dict(color='green', width=2)
    ), row=2, col=1)
    
    fig_rolling.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_rolling, use_container_width=True)

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100

if __name__ == "__main__":
    main()
