"""
Momentum signal calculation using 12-2-1 formation.

12-2-1 Momentum:
- Look back 12 months (252 trading days)
- Skip most recent month (21 trading days) to avoid short-term reversal
- Formation period of 1 month (21 trading days)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

from config.parameters import signal_config
from utils.helpers import calculate_cross_sectional_zscore

logger = logging.getLogger(__name__)


def calculate_momentum_1221(prices: pd.DataFrame,
                           lookback: Optional[int] = None,
                           skip: Optional[int] = None,
                           formation: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate 12-2-1 momentum signal (vectorized).

    Formula:
    Momentum = (Price[t - skip] / Price[t - (lookback + skip)]) - 1

    Args:
        prices: DataFrame with dates as index, tickers as columns
        lookback: Formation period in trading days (default 252 = 12 months)
        skip: Skip period in days (default 21 = 1 month)
        formation: Additional formation period (default 21 = 1 month)

    Returns:
        DataFrame with momentum scores (same shape as prices)
    """
    # Use config defaults if not specified
    lookback = lookback or signal_config.MOMENTUM_LOOKBACK
    skip = skip or signal_config.MOMENTUM_SKIP
    formation = formation or signal_config.MOMENTUM_FORMATION

    logger.info(f"Calculating 12-2-1 momentum (lookback={lookback}, skip={skip}, formation={formation})")

    # Calculate momentum: return from (t-lookback-skip) to (t-skip)
    # This skips the most recent 'skip' days to avoid reversal
    momentum = (prices.shift(skip) / prices.shift(lookback + skip)) - 1

    return momentum


def calculate_simple_momentum(prices: pd.DataFrame,
                              lookback: int = 252) -> pd.DataFrame:
    """
    Calculate simple momentum (no skip period).

    Args:
        prices: DataFrame with prices
        lookback: Lookback period in days

    Returns:
        DataFrame with simple momentum scores
    """
    momentum = (prices / prices.shift(lookback)) - 1
    return momentum


def rank_momentum(momentum_scores: pd.DataFrame,
                 date: str,
                 top_n: Optional[int] = None) -> pd.Series:
    """
    Rank stocks by momentum and select top N.

    Args:
        momentum_scores: DataFrame with momentum scores
        date: Date for ranking
        top_n: Number of top stocks to select (None = return all ranked)

    Returns:
        Series with tickers and their momentum scores (sorted descending)
    """
    if date not in momentum_scores.index:
        logger.warning(f"Date {date} not in momentum scores index")
        return pd.Series()

    # Get momentum scores for this date
    scores = momentum_scores.loc[date].dropna()

    # Sort descending
    ranked = scores.sort_values(ascending=False)

    # Select top N if specified
    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked


def calculate_momentum_dispersion(momentum_scores: pd.DataFrame,
                                 date: str,
                                 universe: Optional[List[str]] = None) -> float:
    """
    Calculate cross-sectional dispersion of momentum.

    High dispersion indicates potential regime change and elevated crash risk.

    Formula:
    Dispersion = StdDev(momentum_scores) / Mean(abs(momentum_scores))

    Args:
        momentum_scores: DataFrame with momentum scores
        date: Date for calculation
        universe: Optional list of tickers to include (None = use all)

    Returns:
        Dispersion metric (float)
    """
    if date not in momentum_scores.index:
        return np.nan

    # Get scores for this date
    scores = momentum_scores.loc[date]

    # Filter to universe if specified
    if universe is not None:
        scores = scores[scores.index.isin(universe)]

    # Drop NaN
    scores = scores.dropna()

    if len(scores) == 0:
        return np.nan

    # Calculate dispersion
    std = scores.std()
    mean_abs = scores.abs().mean()

    if mean_abs == 0:
        return np.nan

    dispersion = std / mean_abs

    return dispersion


def calculate_momentum_zscore(momentum_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional Z-score of momentum.

    For each date, normalize momentum scores across stocks.

    Args:
        momentum_scores: DataFrame with momentum scores

    Returns:
        DataFrame with Z-scored momentum
    """
    logger.info("Calculating cross-sectional momentum Z-scores")
    return calculate_cross_sectional_zscore(momentum_scores)


def identify_momentum_winners(momentum_scores: pd.DataFrame,
                             date: str,
                             percentile: float = 0.8,
                             min_momentum: float = 0.0) -> List[str]:
    """
    Identify momentum winners (top percentile).

    Args:
        momentum_scores: DataFrame with momentum scores
        date: Date for selection
        percentile: Percentile threshold (0.8 = top 20%)
        min_momentum: Minimum momentum value to be considered a winner

    Returns:
        List of ticker symbols
    """
    if date not in momentum_scores.index:
        return []

    scores = momentum_scores.loc[date].dropna()

    # Filter by minimum momentum
    scores = scores[scores >= min_momentum]

    # Get threshold
    threshold = scores.quantile(percentile)

    # Select winners
    winners = scores[scores >= threshold].index.tolist()

    return winners


def identify_momentum_losers(momentum_scores: pd.DataFrame,
                            date: str,
                            percentile: float = 0.2,
                            max_momentum: float = 0.0) -> List[str]:
    """
    Identify momentum losers (bottom percentile).

    Args:
        momentum_scores: DataFrame with momentum scores
        date: Date for selection
        percentile: Percentile threshold (0.2 = bottom 20%)
        max_momentum: Maximum momentum value to be considered a loser

    Returns:
        List of ticker symbols
    """
    if date not in momentum_scores.index:
        return []

    scores = momentum_scores.loc[date].dropna()

    # Filter by maximum momentum
    scores = scores[scores <= max_momentum]

    # Get threshold
    threshold = scores.quantile(percentile)

    # Select losers
    losers = scores[scores <= threshold].index.tolist()

    return losers


def calculate_momentum_strength(momentum_scores: pd.DataFrame,
                               window: int = 60) -> pd.DataFrame:
    """
    Calculate momentum strength (rolling average of momentum).

    This can help identify persistent vs transient momentum.

    Args:
        momentum_scores: DataFrame with momentum scores
        window: Rolling window size in days

    Returns:
        DataFrame with momentum strength
    """
    logger.info(f"Calculating momentum strength (window={window})")
    strength = momentum_scores.rolling(window=window, min_periods=window // 2).mean()
    return strength


def calculate_momentum_trend(momentum_scores: pd.DataFrame,
                            window: int = 60) -> pd.DataFrame:
    """
    Calculate momentum trend (change in momentum).

    Positive trend = momentum is accelerating
    Negative trend = momentum is decelerating

    Args:
        momentum_scores: DataFrame with momentum scores
        window: Window for trend calculation

    Returns:
        DataFrame with momentum trends
    """
    logger.info(f"Calculating momentum trend (window={window})")

    # Calculate change in momentum over window
    trend = momentum_scores - momentum_scores.shift(window)

    return trend


def filter_by_momentum_quality(momentum_scores: pd.DataFrame,
                              prices: pd.DataFrame,
                              date: str,
                              min_price: float = 5.0,
                              min_momentum: float = 0.0) -> pd.Series:
    """
    Filter momentum scores by quality criteria.

    Quality filters:
    - Minimum price (avoid penny stocks)
    - Minimum momentum (positive momentum)
    - No missing data

    Args:
        momentum_scores: DataFrame with momentum scores
        prices: DataFrame with prices
        date: Date for filtering
        min_price: Minimum price threshold
        min_momentum: Minimum momentum threshold

    Returns:
        Series with filtered momentum scores
    """
    if date not in momentum_scores.index or date not in prices.index:
        return pd.Series()

    scores = momentum_scores.loc[date]
    price_levels = prices.loc[date]

    # Filter criteria
    valid = (
        (scores.notna()) &
        (price_levels.notna()) &
        (price_levels >= min_price) &
        (scores >= min_momentum)
    )

    filtered_scores = scores[valid]

    return filtered_scores


def create_momentum_portfolio_weights(momentum_scores: pd.DataFrame,
                                     date: str,
                                     n_positions: int = 50,
                                     weight_method: str = 'equal') -> pd.Series:
    """
    Create portfolio weights based on momentum scores.

    Args:
        momentum_scores: DataFrame with momentum scores
        date: Date for portfolio construction
        n_positions: Number of positions
        weight_method: 'equal' or 'momentum_weighted'

    Returns:
        Series with portfolio weights (sum to 1.0)
    """
    # Select top N by momentum
    ranked = rank_momentum(momentum_scores, date, top_n=n_positions)

    if len(ranked) == 0:
        return pd.Series()

    if weight_method == 'equal':
        # Equal weight
        weights = pd.Series(1.0 / len(ranked), index=ranked.index)

    elif weight_method == 'momentum_weighted':
        # Weight by momentum score (normalized)
        # Shift to positive if any negative values
        scores = ranked.copy()
        if scores.min() < 0:
            scores = scores - scores.min() + 1e-6

        # Normalize to sum to 1
        weights = scores / scores.sum()

    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")

    return weights


def validate_momentum_signals(momentum_scores: pd.DataFrame) -> dict:
    """
    Validate momentum signals for quality and reasonableness.

    Args:
        momentum_scores: DataFrame with momentum scores

    Returns:
        Dictionary with validation metrics
    """
    validation = {
        'total_dates': len(momentum_scores),
        'total_stocks': len(momentum_scores.columns),
        'missing_pct': momentum_scores.isna().sum().sum() / momentum_scores.size * 100,
        'mean_momentum': momentum_scores.mean().mean(),
        'median_momentum': momentum_scores.median().median(),
        'std_momentum': momentum_scores.std().mean(),
        'positive_momentum_pct': (momentum_scores > 0).sum().sum() / momentum_scores.count().sum() * 100,
    }

    # Check for extreme values
    extreme_positive = (momentum_scores > 5.0).sum().sum()  # > 500% return
    extreme_negative = (momentum_scores < -0.9).sum().sum()  # < -90% return

    validation['extreme_positive_count'] = extreme_positive
    validation['extreme_negative_count'] = extreme_negative

    return validation
