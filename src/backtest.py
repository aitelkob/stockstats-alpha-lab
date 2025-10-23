"""
Backtesting module for financial strategies.

This module provides vectorized backtesting with transaction costs,
slippage, and risk controls for evaluating trading strategies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main class for backtesting trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.1
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            max_position_size: Maximum position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        self.results = {}
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        prices: Optional[pd.Series] = None,
        strategy_name: str = "strategy"
    ) -> Dict:
        """
        Run vectorized backtest on signals.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Series with trading signals (-1, 0, 1)
            prices: Series with prices (if None, uses 'close')
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with backtest results
        """
        if prices is None:
            prices = df['close']
        
        # Align signals with prices
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]
        
        # Initialize portfolio
        portfolio = self._initialize_portfolio(len(signals))
        
        # Calculate returns
        returns = self._calculate_returns(prices)
        
        # Run strategy
        portfolio = self._run_strategy(portfolio, signals, prices, returns)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio, strategy_name)
        
        self.results[strategy_name] = results
        logger.info(f"Backtest completed for {strategy_name}")
        
        return results
    
    def _initialize_portfolio(self, length: int) -> pd.DataFrame:
        """Initialize portfolio tracking DataFrame."""
        portfolio = pd.DataFrame(index=range(length))
        portfolio['capital'] = self.initial_capital
        portfolio['position'] = 0.0
        portfolio['shares'] = 0.0
        portfolio['cash'] = self.initial_capital
        portfolio['returns'] = 0.0
        portfolio['cumulative_returns'] = 0.0
        portfolio['drawdown'] = 0.0
        portfolio['max_drawdown'] = 0.0
        
        return portfolio
    
    def _calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate price returns."""
        return prices.pct_change().fillna(0)
    
    def _run_strategy(
        self,
        portfolio: pd.DataFrame,
        signals: pd.Series,
        prices: pd.Series,
        returns: pd.Series
    ) -> pd.DataFrame:
        """Run the trading strategy."""
        for i in range(1, len(portfolio)):
            # Current values
            current_capital = portfolio.loc[i-1, 'capital']
            current_position = portfolio.loc[i-1, 'position']
            current_shares = portfolio.loc[i-1, 'shares']
            current_cash = portfolio.loc[i-1, 'cash']
            
            # Current signal and price
            signal = signals.iloc[i] if i < len(signals) else 0
            price = prices.iloc[i]
            
            # Calculate new position
            target_position = signal * self.max_position_size
            
            # Calculate position change
            position_change = target_position - current_position
            
            if abs(position_change) > 0.001:  # Only trade if significant change
                # Calculate shares to trade
                shares_to_trade = (position_change * current_capital) / price
                
                # Apply slippage
                if shares_to_trade > 0:  # Buying
                    effective_price = price * (1 + self.slippage)
                else:  # Selling
                    effective_price = price * (1 - self.slippage)
                
                # Calculate transaction cost
                transaction_value = abs(shares_to_trade) * effective_price
                transaction_cost = transaction_value * self.commission
                
                # Update shares and cash
                new_shares = current_shares + shares_to_trade
                new_cash = current_cash - (shares_to_trade * effective_price) - transaction_cost
                
                # Ensure we don't go negative on cash
                if new_cash < 0:
                    shares_to_trade = current_cash / effective_price
                    new_shares = current_shares + shares_to_trade
                    new_cash = 0
                
                # Update position
                new_position = (new_shares * price) / current_capital
            else:
                new_shares = current_shares
                new_cash = current_cash
                new_position = current_position
            
            # Calculate portfolio value
            portfolio_value = new_cash + (new_shares * price)
            
            # Calculate returns
            if i > 0:
                portfolio_return = (portfolio_value - current_capital) / current_capital
            else:
                portfolio_return = 0.0
            
            # Update portfolio
            portfolio.loc[i, 'capital'] = portfolio_value
            portfolio.loc[i, 'position'] = new_position
            portfolio.loc[i, 'shares'] = new_shares
            portfolio.loc[i, 'cash'] = new_cash
            portfolio.loc[i, 'returns'] = portfolio_return
            portfolio.loc[i, 'cumulative_returns'] = (portfolio_value / self.initial_capital) - 1
            
            # Calculate drawdown
            running_max = portfolio['cumulative_returns'].iloc[:i+1].max()
            drawdown = running_max - portfolio.loc[i, 'cumulative_returns']
            portfolio.loc[i, 'drawdown'] = drawdown
            portfolio.loc[i, 'max_drawdown'] = max(portfolio['drawdown'].iloc[:i+1])
        
        return portfolio
    
    def _calculate_performance_metrics(
        self,
        portfolio: pd.DataFrame,
        strategy_name: str
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = portfolio['returns'].dropna()
        cumulative_returns = portfolio['cumulative_returns']
        
        # Basic metrics
        total_return = cumulative_returns.iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        max_drawdown = portfolio['max_drawdown'].max()
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Turnover (simplified)
        position_changes = portfolio['position'].diff().abs()
        turnover = position_changes.mean()
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        
        metrics = {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'calmar_ratio': calmar_ratio,
            'turnover': turnover,
            'var_95': var_95,
            'es_95': es_95,
            'num_trades': len(portfolio[portfolio['position'].diff() != 0]),
            'final_capital': portfolio['capital'].iloc[-1],
            'portfolio': portfolio
        }
        
        return metrics


class StrategyBuilder:
    """Helper class for building common trading strategies."""
    
    @staticmethod
    def rsi_trend_strategy(
        df: pd.DataFrame,
        rsi_col: str = "rsi_14",
        trend_col: str = "close_200_sma",
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0
    ) -> pd.Series:
        """
        RSI + Trend filter strategy as mentioned in project plan.
        
        Long when RSI < oversold AND close > trend.
        Short when RSI > overbought AND close < trend.
        """
        signals = pd.Series(0, index=df.index)
        
        # Long condition
        long_condition = (df[rsi_col] < rsi_oversold) & (df['close'] > df[trend_col])
        signals[long_condition] = 1
        
        # Short condition
        short_condition = (df[rsi_col] > rsi_overbought) & (df['close'] < df[trend_col])
        signals[short_condition] = -1
        
        return signals
    
    @staticmethod
    def macd_crossover_strategy(
        df: pd.DataFrame,
        macd_col: str = "macd",
        macd_signal_col: str = "macds"
    ) -> pd.Series:
        """
        MACD crossover strategy.
        
        Long when MACD crosses above signal.
        Short when MACD crosses below signal.
        """
        signals = pd.Series(0, index=df.index)
        
        # MACD crossover
        macd_diff = df[macd_col] - df[macd_signal_col]
        macd_diff_prev = macd_diff.shift(1)
        
        # Long: MACD crosses above signal
        long_condition = (macd_diff > 0) & (macd_diff_prev <= 0)
        signals[long_condition] = 1
        
        # Short: MACD crosses below signal
        short_condition = (macd_diff < 0) & (macd_diff_prev >= 0)
        signals[short_condition] = -1
        
        return signals
    
    @staticmethod
    def bollinger_bands_strategy(
        df: pd.DataFrame,
        price_col: str = "close",
        bb_upper_col: str = "boll_ub",
        bb_lower_col: str = "boll_lb"
    ) -> pd.Series:
        """
        Bollinger Bands mean reversion strategy.
        
        Long when price touches lower band.
        Short when price touches upper band.
        """
        signals = pd.Series(0, index=df.index)
        
        # Long: price at or below lower band
        long_condition = df[price_col] <= df[bb_lower_col]
        signals[long_condition] = 1
        
        # Short: price at or above upper band
        short_condition = df[price_col] >= df[bb_upper_col]
        signals[short_condition] = -1
        
        return signals
    
    @staticmethod
    def volatility_sizing_strategy(
        df: pd.DataFrame,
        base_signals: pd.Series,
        volatility_col: str = "atr_14",
        base_position_size: float = 0.1
    ) -> pd.Series:
        """
        Position sizing based on volatility.
        
        Position size inversely proportional to volatility.
        """
        # Calculate volatility-adjusted position sizes
        volatility = df[volatility_col]
        vol_median = volatility.rolling(20).median()
        
        # Position size inversely proportional to volatility
        vol_adjustment = vol_median / volatility
        vol_adjustment = vol_adjustment.clip(0.5, 2.0)  # Limit adjustment
        
        # Apply to base signals
        sized_signals = base_signals * vol_adjustment * base_position_size
        
        return sized_signals


def run_strategy_comparison(
    df: pd.DataFrame,
    strategies: Dict[str, pd.Series],
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Compare multiple strategies.
    
    Args:
        df: DataFrame with OHLCV data
        strategies: Dictionary mapping strategy names to signals
        initial_capital: Starting capital
        
    Returns:
        DataFrame with comparison results
    """
    engine = BacktestEngine(initial_capital=initial_capital)
    results = []
    
    for strategy_name, signals in strategies.items():
        try:
            result = engine.run_backtest(df, signals, strategy_name=strategy_name)
            
            # Extract key metrics
            metrics = {
                'strategy': strategy_name,
                'total_return': result['total_return'],
                'annualized_return': result['annualized_return'],
                'volatility': result['volatility'],
                'sharpe_ratio': result['sharpe_ratio'],
                'sortino_ratio': result['sortino_ratio'],
                'max_drawdown': result['max_drawdown'],
                'hit_rate': result['hit_rate'],
                'calmar_ratio': result['calmar_ratio'],
                'turnover': result['turnover']
            }
            
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error backtesting {strategy_name}: {e}")
            continue
    
    return pd.DataFrame(results)


def stress_test_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    stress_periods: List[Tuple[str, str]],
    initial_capital: float = 100000.0
) -> Dict[str, Dict]:
    """
    Stress test strategy on specific market periods.
    
    Args:
        df: DataFrame with OHLCV data
        signals: Trading signals
        stress_periods: List of (start_date, end_date) tuples
        initial_capital: Starting capital
        
    Returns:
        Dictionary with stress test results
    """
    engine = BacktestEngine(initial_capital=initial_capital)
    results = {}
    
    for period_name, (start_date, end_date) in stress_periods:
        try:
            # Filter data for stress period
            period_mask = (df.index >= start_date) & (df.index <= end_date)
            period_df = df[period_mask]
            period_signals = signals[period_mask]
            
            if len(period_df) == 0:
                logger.warning(f"No data for period {period_name}")
                continue
            
            # Run backtest
            result = engine.run_backtest(
                period_df, 
                period_signals, 
                strategy_name=f"stress_{period_name}"
            )
            
            results[period_name] = result
            
        except Exception as e:
            logger.error(f"Error stress testing {period_name}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    # Example usage
    from data import DataLoader
    from indicators import add_basic_indicators
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="2y")
    df = add_basic_indicators(df)
    
    # Create strategies
    rsi_signals = StrategyBuilder.rsi_trend_strategy(df)
    macd_signals = StrategyBuilder.macd_crossover_strategy(df)
    
    # Run backtests
    engine = BacktestEngine()
    
    rsi_result = engine.run_backtest(df, rsi_signals, strategy_name="RSI_Trend")
    macd_result = engine.run_backtest(df, macd_signals, strategy_name="MACD_Crossover")
    
    print("RSI Strategy Results:")
    print(f"Total Return: {rsi_result['total_return']:.2%}")
    print(f"Sharpe Ratio: {rsi_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {rsi_result['max_drawdown']:.2%}")
    
    print("\nMACD Strategy Results:")
    print(f"Total Return: {macd_result['total_return']:.2%}")
    print(f"Sharpe Ratio: {macd_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {macd_result['max_drawdown']:.2%}")
