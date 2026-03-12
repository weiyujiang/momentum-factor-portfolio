"""
Combined momentum-value signal generation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from config.parameters import signal_config, portfolio_config
from signals.value import calculate_value_composite

logger = logging.getLogger(__name__)


def generate_combined_signal(momentum_scores: pd.Series,
                            fundamentals: pd.DataFrame,
                            alpha: Optional[float] = None) -> pd.Series:
    """
    Generate combined momentum-value signal.

    Combined = alpha * rank(momentum) + (1-alpha) * rank(value)

    Args:
        momentum_scores: Series with momentum scores for a date
        fundamentals: DataFrame with fundamental data for the same date
        alpha: Weight on momentum (default from config: 0.7)

    Returns:
        Series with combined scores for universe
    """
    alpha = alpha or signal_config.ALPHA

    logger.info(f"Generating combined signal (alpha={alpha:.2f})")

    # Calculate value composite
    value_scores = calculate_value_composite(fundamentals)

    # Get common tickers
    common_tickers = momentum_scores.index.intersection(value_scores.index)

    if len(common_tickers) == 0:
        logger.warning("No common tickers between momentum and value")
        return pd.Series()

    # Filter to common tickers
    mom = momentum_scores[common_tickers]
    val = value_scores[common_tickers]

    # Rank both (percentile rank 0-1)
    mom_rank = mom.rank(pct=True)
    val_rank = val.rank(pct=True)

    # Combined score
    combined = alpha * mom_rank + (1 - alpha) * val_rank

    logger.info(f"Generated combined signals for {len(combined)} stocks")

    return combined


def select_top_stocks(combined_scores: pd.Series,
                     n_positions: Optional[int] = None,
                     min_momentum: float = 0.0,
                     min_value_score: float = 0.0,
                     momentum_scores: Optional[pd.Series] = None,
                     value_scores: Optional[pd.Series] = None) -> List[str]:
    """
    Select top stocks based on combined signal with filters.

    Args:
        combined_scores: Series with combined scores
        n_positions: Number of positions to select
        min_momentum: Minimum momentum score filter
        min_value_score: Minimum value score filter
        momentum_scores: Original momentum scores (for filtering)
        value_scores: Original value scores (for filtering)

    Returns:
        List of selected ticker symbols
    """
    n_positions = n_positions or portfolio_config.LONG_POSITIONS

    # Start with all scores
    filtered_scores = combined_scores.copy()

    # Apply momentum filter if provided
    if min_momentum > 0 and momentum_scores is not None:
        momentum_filter = momentum_scores >= min_momentum
        filtered_scores = filtered_scores[momentum_filter]

    # Apply value filter if provided
    if min_value_score != 0.0 and value_scores is not None:
        value_filter = value_scores >= min_value_score
        filtered_scores = filtered_scores[value_filter]

    # Sort by combined score descending
    sorted_scores = filtered_scores.sort_values(ascending=False)

    # Select top N
    selected = sorted_scores.head(n_positions).index.tolist()

    logger.info(f"Selected {len(selected)} stocks from {len(filtered_scores)} candidates")

    return selected


def create_quintile_portfolios(combined_scores: pd.Series,
                              n_quintiles: int = 5) -> dict:
    """
    Create quintile portfolios based on combined scores.

    Useful for analyzing signal strength across quintiles.

    Args:
        combined_scores: Series with combined scores
        n_quintiles: Number of quintiles (default 5)

    Returns:
        Dictionary with {quintile: [tickers]}
    """
    # Drop NaN
    scores = combined_scores.dropna()

    # Calculate quintile thresholds
    quintiles = {}
    scores_sorted = scores.sort_values()

    for q in range(1, n_quintiles + 1):
        start_idx = int((q - 1) / n_quintiles * len(scores_sorted))
        end_idx = int(q / n_quintiles * len(scores_sorted))

        quintile_tickers = scores_sorted.iloc[start_idx:end_idx].index.tolist()
        quintiles[f'Q{q}'] = quintile_tickers

    return quintiles


def calculate_signal_correlation(momentum_scores: pd.Series,
                                value_scores: pd.Series) -> float:
    """
    Calculate correlation between momentum and value signals.

    High negative correlation is typical (momentum stocks = growth = low value).

    Args:
        momentum_scores: Momentum scores
        value_scores: Value scores

    Returns:
        Correlation coefficient
    """
    # Get common tickers
    common = momentum_scores.index.intersection(value_scores.index)

    if len(common) < 10:
        return np.nan

    corr = momentum_scores[common].corr(value_scores[common])

    return corr


def validate_combined_signals(combined_scores: pd.Series,
                             momentum_scores: pd.Series,
                             value_scores: pd.Series) -> dict:
    """
    Validate combined signal quality.

    Args:
        combined_scores: Combined scores
        momentum_scores: Momentum scores
        value_scores: Value scores

    Returns:
        Dictionary with validation metrics
    """
    validation = {
        'total_stocks': len(combined_scores),
        'mean_score': combined_scores.mean(),
        'median_score': combined_scores.median(),
        'std_score': combined_scores.std(),
    }

    # Calculate signal correlation
    validation['momentum_value_corr'] = calculate_signal_correlation(
        momentum_scores,
        value_scores
    )

    # Check for sufficient spread
    validation['score_range'] = combined_scores.max() - combined_scores.min()

    # Check top decile
    top_decile_threshold = combined_scores.quantile(0.9)
    validation['top_decile_count'] = (combined_scores >= top_decile_threshold).sum()

    return validation
