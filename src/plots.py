"""
Plotting and visualization module for financial analysis.

This module provides comprehensive plotting functions for technical indicators,
backtest results, and performance analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class Plotter:
    """Main class for creating financial plots and reports."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize plotter.

        Args:
            figsize: Default figure size
            dpi: DPI for plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            "price": "#1f77b4",
            "volume": "#ff7f0e",
            "indicator": "#2ca02c",
            "signal": "#d62728",
            "profit": "#2ca02c",
            "loss": "#d62728",
            "neutral": "#7f7f7f",
        }

    def plot_price_and_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str],
        title: str = "Price and Technical Indicators",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot price with technical indicators.

        Args:
            df: DataFrame with OHLCV and indicator data
            indicators: List of indicator column names to plot
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(
            len(indicators) + 1, 1, figsize=self.figsize, dpi=self.dpi
        )
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Plot price
        axes[0].plot(
            df.index,
            df["close"],
            color=self.colors["price"],
            linewidth=2,
            label="Close Price",
        )
        axes[0].set_ylabel("Price ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot indicators
        for i, indicator in enumerate(indicators):
            if indicator in df.columns:
                axes[i + 1].plot(
                    df.index,
                    df[indicator],
                    color=self.colors["indicator"],
                    linewidth=1.5,
                )
                axes[i + 1].set_ylabel(indicator.replace("_", " ").title())
                axes[i + 1].grid(True, alpha=0.3)
            else:
                logger.warning(f"Indicator {indicator} not found in DataFrame")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_backtest_results(
        self,
        backtest_results: Dict,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Backtest Results",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comprehensive backtest results.

        Args:
            backtest_results: Dictionary with backtest results
            benchmark_returns: Benchmark returns for comparison
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        portfolio = backtest_results["portfolio"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Equity curve
        axes[0, 0].plot(
            portfolio.index,
            portfolio["cumulative_returns"],
            color=self.colors["price"],
            linewidth=2,
            label="Strategy",
        )

        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            axes[0, 0].plot(
                portfolio.index,
                benchmark_cumulative,
                color=self.colors["neutral"],
                linewidth=2,
                label="Benchmark",
            )

        axes[0, 0].set_title("Cumulative Returns")
        axes[0, 0].set_ylabel("Cumulative Return")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Drawdown
        axes[0, 1].fill_between(
            portfolio.index,
            portfolio["drawdown"],
            0,
            color=self.colors["loss"],
            alpha=0.7,
            label="Drawdown",
        )
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].set_ylabel("Drawdown")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Returns distribution
        returns = portfolio["returns"].dropna()
        axes[1, 0].hist(
            returns,
            bins=50,
            alpha=0.7,
            color=self.colors["indicator"],
            edgecolor="black",
        )
        axes[1, 0].axvline(
            returns.mean(),
            color=self.colors["signal"],
            linestyle="--",
            label=f"Mean: {returns.mean():.4f}",
        )
        axes[1, 0].set_title("Returns Distribution")
        axes[1, 0].set_xlabel("Daily Return")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Rolling Sharpe ratio
        rolling_sharpe = (
            returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        )
        axes[1, 1].plot(
            portfolio.index,
            rolling_sharpe,
            color=self.colors["indicator"],
            linewidth=1.5,
        )
        axes[1, 1].set_title("Rolling Sharpe Ratio (252 days)")
        axes[1, 1].set_ylabel("Sharpe Ratio")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Backtest plot saved to {save_path}")

        return fig

    def plot_strategy_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        title: str = "Strategy Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot strategy comparison heatmap.

        Args:
            comparison_df: DataFrame with strategy comparison results
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["total_return", "sharpe_ratio", "max_drawdown", "hit_rate"]

        # Select and normalize metrics
        plot_data = comparison_df.set_index("strategy")[metrics]

        # Normalize for heatmap (0-1 scale)
        normalized_data = (plot_data - plot_data.min()) / (
            plot_data.max() - plot_data.min()
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(normalized_data.T, cmap="RdYlGn", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(plot_data.index)))
        ax.set_xticklabels(plot_data.index, rotation=45, ha="right")
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m.replace("_", " ").title() for m in metrics])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Score", rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(plot_data.index)):
            for j in range(len(metrics)):
                text = ax.text(
                    i,
                    j,
                    f"{plot_data.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            feature_importance: Dictionary mapping features to importance scores
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        top_features = sorted_features[:top_n]

        features, importances = zip(*top_features)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)

        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color=self.colors["indicator"], alpha=0.7)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace("_", " ").title() for f in features])
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.3f}",
                ha="left",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Heatmap",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot correlation heatmap for indicators.

        Args:
            df: DataFrame with indicator data
            columns: List of columns to include
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if columns is None:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = [
                col
                for col in numeric_cols
                if col not in ["open", "high", "low", "close", "volume"]
            ]

        # Calculate correlation matrix
        corr_matrix = df[columns].corr()

        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(
            [col.replace("_", " ").title() for col in columns], rotation=45, ha="right"
        )
        ax.set_yticklabels([col.replace("_", " ").title() for col in columns])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation", rotation=270, labelpad=20)

        # Add correlation values
        for i in range(len(columns)):
            for j in range(len(columns)):
                text = ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Correlation plot saved to {save_path}")

        return fig

    def plot_monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot monthly returns as a heatmap.

        Args:
            returns: Series of returns
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        # Convert to monthly returns
        monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns["year"] = monthly_returns.index.year
        monthly_returns["month"] = monthly_returns.index.month

        pivot_table = monthly_returns.pivot(index="year", columns="month", values=0)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(pivot_table.values, cmap="RdYlGn", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Monthly Return", rotation=270, labelpad=20)

        # Add value annotations
        for i in range(len(pivot_table.index)):
            for j in range(12):
                if not pd.isna(pivot_table.iloc[i, j]):
                    text = ax.text(
                        j,
                        i,
                        f"{pivot_table.iloc[i, j]:.1%}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Monthly returns plot saved to {save_path}")

        return fig

    def create_tearsheet(
        self,
        backtest_results: Dict,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive tearsheet.

        Args:
            backtest_results: Dictionary with backtest results
            benchmark_returns: Benchmark returns for comparison
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        portfolio = backtest_results["portfolio"]
        returns = portfolio["returns"].dropna()

        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            portfolio.index,
            portfolio["cumulative_returns"],
            color=self.colors["price"],
            linewidth=2,
            label="Strategy",
        )

        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            ax1.plot(
                portfolio.index,
                benchmark_cumulative,
                color=self.colors["neutral"],
                linewidth=2,
                label="Benchmark",
            )

        ax1.set_title("Cumulative Returns", fontweight="bold")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.fill_between(
            portfolio.index,
            portfolio["drawdown"],
            0,
            color=self.colors["loss"],
            alpha=0.7,
        )
        ax2.set_title("Drawdown", fontweight="bold")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)

        # Returns distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(
            returns,
            bins=50,
            alpha=0.7,
            color=self.colors["indicator"],
            edgecolor="black",
        )
        ax3.axvline(
            returns.mean(),
            color=self.colors["signal"],
            linestyle="--",
            label=f"Mean: {returns.mean():.4f}",
        )
        ax3.set_title("Returns Distribution", fontweight="bold")
        ax3.set_xlabel("Daily Return")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Rolling Sharpe
        ax4 = fig.add_subplot(gs[1, 1])
        rolling_sharpe = (
            returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        )
        ax4.plot(
            portfolio.index,
            rolling_sharpe,
            color=self.colors["indicator"],
            linewidth=1.5,
        )
        ax4.set_title("Rolling Sharpe (252d)", fontweight="bold")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.grid(True, alpha=0.3)

        # Monthly returns heatmap
        ax5 = fig.add_subplot(gs[1, 2])
        monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns["year"] = monthly_returns.index.year
        monthly_returns["month"] = monthly_returns.index.month

        pivot_table = monthly_returns.pivot(index="year", columns="month", values=0)
        im = ax5.imshow(pivot_table.values, cmap="RdYlGn", aspect="auto")
        ax5.set_title("Monthly Returns", fontweight="bold")
        ax5.set_xlabel("Month")
        ax5.set_ylabel("Year")

        # Performance metrics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        metrics_text = f"""
        Performance Metrics:
        Total Return: {backtest_results['total_return']:.2%}
        Annualized Return: {backtest_results['annualized_return']:.2%}
        Volatility: {backtest_results['volatility']:.2%}
        Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}
        Sortino Ratio: {backtest_results['sortino_ratio']:.2f}
        Max Drawdown: {backtest_results['max_drawdown']:.2%}
        Hit Rate: {backtest_results['hit_rate']:.2%}
        Calmar Ratio: {backtest_results['calmar_ratio']:.2f}
        """

        ax6.text(
            0.1,
            0.5,
            metrics_text,
            transform=ax6.transAxes,
            fontsize=12,
            verticalalignment="center",
            fontfamily="monospace",
        )

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Tearsheet saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage
    from backtest import BacktestEngine, StrategyBuilder
    from data import DataLoader
    from indicators import add_basic_indicators

    # Load and prepare data
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="1y")
    df = add_basic_indicators(df)

    # Create strategy and backtest
    signals = StrategyBuilder.rsi_trend_strategy(df)
    engine = BacktestEngine()
    results = engine.run_backtest(df, signals, strategy_name="RSI_Trend")

    # Create plots
    plotter = Plotter()

    # Plot price and indicators
    indicators = ["rsi_14", "close_20_sma", "macd"]
    fig1 = plotter.plot_price_and_indicators(df, indicators)

    # Plot backtest results
    fig2 = plotter.plot_backtest_results(results)

    # Create tearsheet
    fig3 = plotter.create_tearsheet(results)

    plt.show()
