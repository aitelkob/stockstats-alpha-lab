"""
Data loading and preprocessing module for stockstats alpha lab.

This module provides clean interfaces for loading OHLCV data from various sources
and ensuring data quality for technical analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd
import polars as pl
import yfinance as yf
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Main data loading class with support for multiple data sources."""
    
    def __init__(self, source: str = "yfinance"):
        """
        Initialize data loader.
        
        Args:
            source: Data source ('yfinance', 'polars', 'pandas')
        """
        self.source = source
        self.required_columns = ["open", "high", "low", "close", "volume"]
    
    def load_single_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (if None, uses period)
            end_date: End date (if None, uses today)
            period: Period if start_date/end_date not provided
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        if self.source == "yfinance":
            return self._load_yfinance(ticker, start_date, end_date, period)
        else:
            raise ValueError(f"Source {self.source} not implemented")
    
    def load_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "2y"
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (if None, uses period)
            end_date: End date (if None, uses today)
            period: Period if start_date/end_date not provided
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data = {}
        
        for ticker in tqdm(tickers, desc="Loading tickers"):
            try:
                df = self.load_single_ticker(ticker, start_date, end_date, period)
                data[ticker] = df
                logger.info(f"Loaded {ticker}: {len(df)} records")
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")
                continue
        
        return data
    
    def _load_yfinance(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """Load data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Ensure proper column names (yfinance uses title case)
            df.columns = df.columns.str.lower()
            
            # Validate required columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Clean data
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with NaN values in OHLCV
        df = df.dropna(subset=self.required_columns)
        
        # Ensure positive values
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Ensure high >= low, high >= open, high >= close
        df = df[df["high"] >= df["low"]]
        df = df[df["high"] >= df["open"]]
        df = df[df["high"] >= df["close"]]
        
        # Ensure low <= open, low <= close
        df = df[df["low"] <= df["open"]]
        df = df[df["low"] <= df["close"]]
        
        # Sort by date
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Cleaned data: {len(df)} records remaining")
        return df
    
    def to_polars(self, df: pd.DataFrame) -> pl.DataFrame:
        """Convert pandas DataFrame to Polars DataFrame."""
        return pl.from_pandas(df)
    
    def from_polars(self, df: pl.DataFrame) -> pd.DataFrame:
        """Convert Polars DataFrame to pandas DataFrame."""
        return df.to_pandas()


def get_sample_tickers() -> List[str]:
    """Get a sample list of tickers for demonstration."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # Tech giants
        "JPM", "BAC", "WFC", "GS", "MS",          # Financials
        "JNJ", "PFE", "UNH", "ABBV", "MRK",       # Healthcare
        "XOM", "CVX", "COP", "EOG", "SLB",        # Energy
        "SPY", "QQQ", "IWM", "VTI", "VEA"         # ETFs
    ]


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return metrics.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "total_records": len(df),
        "date_range": (df.index.min(), df.index.max()),
        "missing_values": df.isnull().sum().to_dict(),
        "zero_volume_days": (df["volume"] == 0).sum(),
        "negative_prices": ((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum(),
        "ohlc_violations": 0,  # Will calculate below
    }
    
    # Check OHLC relationships
    ohlc_violations = 0
    ohlc_violations += (df["high"] < df["low"]).sum()
    ohlc_violations += (df["high"] < df["open"]).sum()
    ohlc_violations += (df["high"] < df["close"]).sum()
    ohlc_violations += (df["low"] > df["open"]).sum()
    ohlc_violations += (df["low"] > df["close"]).sum()
    
    metrics["ohlc_violations"] = ohlc_violations
    
    return metrics


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load single ticker
    df = loader.load_single_ticker("AAPL", period="1y")
    print(f"AAPL data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Validate data quality
    quality = validate_data_quality(df)
    print(f"Data quality metrics: {quality}")
    
    # Load multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = loader.load_multiple_tickers(tickers, period="6mo")
    print(f"Loaded {len(data)} tickers")
