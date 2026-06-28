"""
data.py
Data pipeline for Neural Latent Factor Discovery Engine
Handles CSV loading, yfinance download, log returns, and standardization.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import yfinance as yf


def load_prices_csv(path: str) -> pd.DataFrame:
    """Load historical prices from a CSV file."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def download_prices_yf(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download historical prices from Yahoo Finance."""
    df = yf.download(tickers, start=start, end=end)
    # Handle both single and multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: columns like ('Close', 'AAPL')
        prices = df['Close']
    else:
        # Single ticker: columns like 'Open', 'Close', etc.
        prices = df['Close'].to_frame()
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Standardize returns (zero mean, unit variance)."""
    return (returns - returns.mean()) / returns.std()


def get_data(source: str, path_or_tickers, start: Optional[str]=None, end: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price data and compute log/standardized returns.
    source: 'csv' or 'yfinance'
    path_or_tickers: CSV path or list of tickers
    Returns: (log_returns, standardized_returns)
    """
    if source == 'csv':
        prices = load_prices_csv(path_or_tickers)
    elif source == 'yfinance':
        prices = download_prices_yf(path_or_tickers, start, end)
    else:
        raise ValueError("Unknown data source.")
    log_ret = compute_log_returns(prices)
    std_ret = standardize_returns(log_ret)
    return log_ret, std_ret
