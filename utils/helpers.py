"""
Common utility functions used across the project.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta


def get_trading_days(start: str, end: str, freq: str = 'D') -> pd.DatetimeIndex:
    """
    Get trading days (excluding weekends/holidays).

    Args:
        start: Start date ('YYYY-MM-DD')
        end: End date ('YYYY-MM-DD')
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
        DatetimeIndex with trading days
    """
    # Create date range
    date_range = pd.date_range(start=start, end=end, freq=freq)

    # Filter out weekends (Saturday=5, Sunday=6)
    if freq == 'D':
        date_range = date_range[date_range.weekday < 5]

    return date_range


def align_dataframes(*dfs: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Align multiple dataframes to common date index.

    Args:
        *dfs: Variable number of DataFrames

    Returns:
        List of aligned DataFrames
    """
    if len(dfs) == 0:
        return []

    # Get common dates (intersection of all date indices)
    common_dates = dfs[0].index
    for df in dfs[1:]:
        common_dates = common_dates.intersection(df.index)

    # Reindex all dataframes to common dates
    aligned = [df.loc[common_dates] for df in dfs]

    return aligned


def normalize_tickers(tickers: List[str]) -> List[str]:
    """
    Normalize ticker symbols (handle special characters).

    Args:
        tickers: List of ticker symbols

    Returns:
        List of normalized ticker symbols
    """
    normalized = []
    for ticker in tickers:
        # Remove whitespace
        ticker = ticker.strip().upper()

        # Handle common issues
        # BRK.B -> BRK-B (yfinance format)
        ticker = ticker.replace('.', '-')

        normalized.append(ticker)

    return normalized


def calculate_rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling Z-score.

    Args:
        series: Input time series
        window: Rolling window size (default 252 trading days = 1 year)

    Returns:
        Rolling Z-score series
    """
    rolling_mean = series.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window=window, min_periods=window // 2).std()

    zscore = (series - rolling_mean) / rolling_std
    return zscore


def calculate_cross_sectional_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional Z-score (across columns for each row).

    Args:
        data: DataFrame with dates as index, tickers as columns

    Returns:
        DataFrame with Z-scores
    """
    # Calculate mean and std across columns for each row
    mean = data.mean(axis=1)
    std = data.std(axis=1)

    # Broadcast and calculate Z-score
    zscore = data.sub(mean, axis=0).div(std, axis=0)

    return zscore


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize returns from any frequency.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized return
    """
    # Compound returns
    total_return = (1 + returns).prod() - 1

    # Annualize
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0

    years = n_periods / periods_per_year
    annualized = (1 + total_return) ** (1 / years) - 1

    return annualized


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize volatility.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_compound_return(returns: pd.Series) -> float:
    """
    Calculate compound return from series of returns.

    Args:
        returns: Series of returns

    Returns:
        Total compound return
    """
    return (1 + returns).prod() - 1


def winsorize(data: pd.Series, lower_percentile: float = 0.01,
              upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize data by capping extreme values.

    Args:
        data: Input series
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold

    Returns:
        Winsorized series
    """
    lower_bound = data.quantile(lower_percentile)
    upper_bound = data.quantile(upper_percentile)

    return data.clip(lower=lower_bound, upper=upper_bound)


def get_month_end_dates(start: str, end: str) -> pd.DatetimeIndex:
    """
    Get month-end dates between start and end.

    Args:
        start: Start date ('YYYY-MM-DD')
        end: End date ('YYYY-MM-DD')

    Returns:
        DatetimeIndex with month-end dates
    """
    date_range = pd.date_range(start=start, end=end, freq='M')
    return date_range


def forward_fill_limited(data: pd.DataFrame, max_periods: int = 5) -> pd.DataFrame:
    """
    Forward fill missing data with a maximum limit.

    Args:
        data: DataFrame with potential missing values
        max_periods: Maximum number of periods to forward fill

    Returns:
        DataFrame with limited forward filling
    """
    return data.fillna(method='ffill', limit=max_periods)


def remove_outliers(data: pd.Series, n_std: float = 5.0) -> pd.Series:
    """
    Remove outliers beyond n standard deviations.

    Args:
        data: Input series
        n_std: Number of standard deviations for outlier threshold

    Returns:
        Series with outliers replaced by NaN
    """
    mean = data.mean()
    std = data.std()

    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std

    cleaned = data.copy()
    cleaned[(cleaned < lower_bound) | (cleaned > upper_bound)] = np.nan

    return cleaned


def rank_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rank normalize data (convert to percentile ranks 0-1).

    Args:
        data: DataFrame to normalize

    Returns:
        DataFrame with rank-normalized values (0-1)
    """
    return data.rank(axis=1, pct=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series,
                fill_value: float = np.nan) -> pd.Series:
    """
    Safely divide two series, handling division by zero.

    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero

    Returns:
        Result of division
    """
    result = numerator / denominator
    result[denominator == 0] = fill_value
    return result


def format_large_number(num: float) -> str:
    """
    Format large numbers for display (e.g., 1,234,567 -> 1.23M).

    Args:
        num: Number to format

    Returns:
        Formatted string
    """
    if abs(num) >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"
