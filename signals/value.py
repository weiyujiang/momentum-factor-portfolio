"""
Value signal calculation from fundamental data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from utils.helpers import calculate_cross_sectional_zscore

logger = logging.getLogger(__name__)


def calculate_value_composite(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Calculate composite value score.

    Composite = Z-score(B/M) + Z-score(E/P)

    Higher composite = more value (cheaper stock)

    Args:
        fundamentals: DataFrame with columns [B/M, E/P, ...]

    Returns:
        Series with composite value scores
    """
    logger.info("Calculating composite value scores")

    if 'B/M' not in fundamentals.columns or 'E/P' not in fundamentals.columns:
        logger.error("Missing B/M or E/P columns in fundamentals")
        return pd.Series()

    # Calculate Z-scores for each metric
    bm_zscore = (fundamentals['B/M'] - fundamentals['B/M'].mean()) / fundamentals['B/M'].std()
    ep_zscore = (fundamentals['E/P'] - fundamentals['E/P'].mean()) / fundamentals['E/P'].std()

    # Composite (equal weight)
    composite = bm_zscore + ep_zscore

    return composite


def rank_value(value_scores: pd.Series, top_n: Optional[int] = None) -> pd.Series:
    """
    Rank stocks by value and select top N.

    Args:
        value_scores: Series with value scores
        top_n: Number of top stocks (None = return all)

    Returns:
        Series with top value stocks (sorted descending)
    """
    # Drop NaN
    scores = value_scores.dropna()

    # Sort descending (higher value score = cheaper = better)
    ranked = scores.sort_values(ascending=False)

    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked


def calculate_value_spread(fundamentals: pd.DataFrame,
                          top_percentile: int = 20,
                          bottom_percentile: int = 20) -> float:
    """
    Calculate value spread between expensive and cheap stocks.

    Spread = Mean(B/M, top 20%) - Mean(B/M, bottom 20%)

    High spread indicates elevated crash risk (bubble conditions).

    Args:
        fundamentals: DataFrame with fundamental data
        top_percentile: Top percentile for "value" stocks
        bottom_percentile: Bottom percentile for "growth" stocks

    Returns:
        Value spread (float)
    """
    if 'B/M' not in fundamentals.columns:
        return np.nan

    bm = fundamentals['B/M'].dropna()

    if len(bm) < 10:
        return np.nan

    # Calculate percentile thresholds
    top_threshold = np.percentile(bm, 100 - top_percentile)
    bottom_threshold = np.percentile(bm, bottom_percentile)

    # Get mean B/M for value vs growth
    value_bm = bm[bm >= top_threshold].mean()
    growth_bm = bm[bm <= bottom_threshold].mean()

    # Calculate spread
    spread = value_bm - growth_bm

    return spread


def identify_value_stocks(fundamentals: pd.DataFrame,
                         percentile: float = 0.7,
                         min_bm: float = 0.0,
                         min_ep: float = 0.0) -> List[str]:
    """
    Identify value stocks (top percentile by composite score).

    Args:
        fundamentals: DataFrame with fundamental data
        percentile: Percentile threshold (0.7 = top 30%)
        min_bm: Minimum B/M ratio
        min_ep: Minimum E/P ratio

    Returns:
        List of ticker symbols
    """
    # Calculate composite
    composite = calculate_value_composite(fundamentals)

    # Apply minimum filters
    valid = composite.copy()

    if 'B/M' in fundamentals.columns:
        valid = valid[fundamentals['B/M'] >= min_bm]

    if 'E/P' in fundamentals.columns:
        valid = valid[fundamentals['E/P'] >= min_ep]

    # Get threshold
    if len(valid) == 0:
        return []

    threshold = valid.quantile(percentile)

    # Select value stocks
    value_stocks = valid[valid >= threshold].index.tolist()

    return value_stocks


def filter_momentum_by_value(momentum_scores: pd.Series,
                             fundamentals: pd.DataFrame,
                             value_percentile: float = 0.5) -> pd.Series:
    """
    Filter momentum scores to exclude expensive stocks.

    This implements value-hedged momentum by removing growth/expensive stocks.

    Args:
        momentum_scores: Series with momentum scores
        fundamentals: DataFrame with fundamental data
        value_percentile: Percentile cutoff (0.5 = exclude bottom 50% by value)

    Returns:
        Series with filtered momentum scores
    """
    # Calculate value composite
    value_scores = calculate_value_composite(fundamentals)

    # Get value threshold
    threshold = value_scores.quantile(1 - value_percentile)

    # Filter momentum to stocks above value threshold
    filtered = momentum_scores[value_scores >= threshold]

    return filtered


def calculate_value_momentum_score(momentum_scores: pd.Series,
                                  fundamentals: pd.DataFrame,
                                  momentum_weight: float = 0.7) -> pd.Series:
    """
    Calculate combined value-momentum score.

    Score = momentum_weight * rank(momentum) + (1 - momentum_weight) * rank(value)

    Args:
        momentum_scores: Series with momentum scores
        fundamentals: DataFrame with fundamental data
        momentum_weight: Weight on momentum (default 0.7)

    Returns:
        Series with combined scores
    """
    # Calculate value composite
    value_scores = calculate_value_composite(fundamentals)

    # Get common tickers
    common_tickers = momentum_scores.index.intersection(value_scores.index)

    if len(common_tickers) == 0:
        logger.warning("No common tickers between momentum and value scores")
        return pd.Series()

    # Rank both signals (percentile rank 0-1)
    momentum_rank = momentum_scores[common_tickers].rank(pct=True)
    value_rank = value_scores[common_tickers].rank(pct=True)

    # Combined score
    combined = (momentum_weight * momentum_rank +
                (1 - momentum_weight) * value_rank)

    return combined


def validate_fundamental_data(fundamentals: pd.DataFrame) -> dict:
    """
    Validate fundamental data quality.

    Args:
        fundamentals: DataFrame with fundamental data

    Returns:
        Dictionary with validation metrics
    """
    validation = {
        'total_stocks': len(fundamentals),
        'has_bm': fundamentals['B/M'].notna().sum() if 'B/M' in fundamentals.columns else 0,
        'has_ep': fundamentals['E/P'].notna().sum() if 'E/P' in fundamentals.columns else 0,
        'has_market_cap': fundamentals['market_cap'].notna().sum() if 'market_cap' in fundamentals.columns else 0,
    }

    if 'B/M' in fundamentals.columns:
        bm = fundamentals['B/M'].dropna()
        if len(bm) > 0:
            validation['bm_mean'] = bm.mean()
            validation['bm_median'] = bm.median()
            validation['bm_std'] = bm.std()

    if 'E/P' in fundamentals.columns:
        ep = fundamentals['E/P'].dropna()
        if len(ep) > 0:
            validation['ep_mean'] = ep.mean()
            validation['ep_median'] = ep.median()
            validation['ep_std'] = ep.std()

    # Coverage
    if len(fundamentals) > 0:
        validation['bm_coverage'] = validation['has_bm'] / len(fundamentals) * 100
        validation['ep_coverage'] = validation['has_ep'] / len(fundamentals) * 100

    return validation
