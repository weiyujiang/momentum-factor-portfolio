"""
Portfolio construction with various weighting schemes.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from config.parameters import portfolio_config, risk_config

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Portfolio construction with various weighting schemes."""

    def __init__(self, weight_method: Optional[str] = None):
        """
        Initialize PortfolioConstructor.

        Args:
            weight_method: Weighting method ('equal', 'market_cap', 'signal', 'risk_parity')
        """
        self.weight_method = weight_method or portfolio_config.WEIGHT_METHOD

    def construct_long_portfolio(self,
                                selected_tickers: List[str],
                                fundamentals: Optional[pd.DataFrame] = None,
                                prices: Optional[pd.DataFrame] = None,
                                signals: Optional[pd.Series] = None,
                                max_position: Optional[float] = None) -> pd.Series:
        """
        Construct long portfolio weights.

        Args:
            selected_tickers: List of selected stocks
            fundamentals: DataFrame with fundamental data (for market cap)
            prices: DataFrame with prices (for volatility weighting)
            signals: Series with signal strength (for signal weighting)
            max_position: Maximum position size

        Returns:
            Series with tickers as index, weights as values (sum to 1.0)
        """
        max_position = max_position or portfolio_config.MAX_POSITION_SIZE

        if len(selected_tickers) == 0:
            return pd.Series()

        if self.weight_method == 'equal':
            weights = self._equal_weight(selected_tickers)

        elif self.weight_method == 'market_cap':
            weights = self._market_cap_weight(selected_tickers, fundamentals)

        elif self.weight_method == 'signal':
            weights = self._signal_weight(selected_tickers, signals)

        elif self.weight_method == 'risk_parity':
            weights = self._risk_parity_weight(selected_tickers, prices)

        else:
            logger.warning(f"Unknown weight method: {self.weight_method}, using equal weight")
            weights = self._equal_weight(selected_tickers)

        # Apply constraints
        weights = self.ensure_constraints(weights, max_position)

        return weights

    def _equal_weight(self, tickers: List[str]) -> pd.Series:
        """Equal weight portfolio."""
        weight = 1.0 / len(tickers)
        weights = pd.Series(weight, index=tickers)
        return weights

    def _market_cap_weight(self,
                          tickers: List[str],
                          fundamentals: Optional[pd.DataFrame]) -> pd.Series:
        """Market cap weighted portfolio."""
        if fundamentals is None or 'market_cap' not in fundamentals.columns:
            logger.warning("No market cap data, falling back to equal weight")
            return self._equal_weight(tickers)

        # Get market caps for selected tickers
        market_caps = fundamentals.loc[fundamentals.index.isin(tickers), 'market_cap']
        market_caps = market_caps.dropna()

        if len(market_caps) == 0:
            return self._equal_weight(tickers)

        # Normalize to sum to 1
        weights = market_caps / market_caps.sum()

        return weights

    def _signal_weight(self,
                      tickers: List[str],
                      signals: Optional[pd.Series]) -> pd.Series:
        """Signal strength weighted portfolio."""
        if signals is None:
            logger.warning("No signals provided, falling back to equal weight")
            return self._equal_weight(tickers)

        # Get signals for selected tickers
        ticker_signals = signals[signals.index.isin(tickers)]
        ticker_signals = ticker_signals.dropna()

        if len(ticker_signals) == 0:
            return self._equal_weight(tickers)

        # Shift to positive if needed
        if ticker_signals.min() < 0:
            ticker_signals = ticker_signals - ticker_signals.min() + 1e-6

        # Normalize to sum to 1
        weights = ticker_signals / ticker_signals.sum()

        return weights

    def _risk_parity_weight(self,
                           tickers: List[str],
                           prices: Optional[pd.DataFrame]) -> pd.Series:
        """Risk parity (inverse volatility) weighted portfolio."""
        if prices is None:
            logger.warning("No price data, falling back to equal weight")
            return self._equal_weight(tickers)

        # Calculate volatilities
        returns = prices[tickers].pct_change()
        volatilities = returns.std()
        volatilities = volatilities.dropna()

        if len(volatilities) == 0 or volatilities.min() == 0:
            return self._equal_weight(tickers)

        # Inverse volatility weights
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()

        return weights

    def apply_hedge(self,
                   long_weights: pd.Series,
                   hedge_allocation: float,
                   hedge_method: str = 'cash') -> pd.Series:
        """
        Apply crash hedge to portfolio.

        Methods:
        - 'cash': Reduce exposure to (1 - hedge_allocation)
        - 'puts': Allocate to put options (modeled as cash drag)
        - 'inverse': Allocate to inverse ETF

        Args:
            long_weights: Long portfolio weights
            hedge_allocation: Hedge allocation (0-1)
            hedge_method: Hedging method

        Returns:
            Adjusted portfolio weights
        """
        if hedge_allocation <= 0:
            return long_weights

        # Clamp hedge allocation
        hedge_allocation = min(hedge_allocation, risk_config.HEDGE_ALLOCATION_MAX)

        if hedge_method == 'cash':
            # Scale down long positions
            adjusted_weights = long_weights * (1 - hedge_allocation)

        elif hedge_method == 'puts':
            # Model puts as cash drag (simplified)
            adjusted_weights = long_weights * (1 - hedge_allocation)

        else:
            logger.warning(f"Unknown hedge method: {hedge_method}, using cash")
            adjusted_weights = long_weights * (1 - hedge_allocation)

        return adjusted_weights

    def ensure_constraints(self,
                          weights: pd.Series,
                          max_position: float,
                          min_position: Optional[float] = None) -> pd.Series:
        """
        Apply position size constraints.

        Args:
            weights: Portfolio weights
            max_position: Maximum position size
            min_position: Minimum position size

        Returns:
            Constrained weights
        """
        min_position = min_position or portfolio_config.MIN_POSITION_SIZE

        # Clip weights
        weights = weights.clip(lower=min_position, upper=max_position)

        # Renormalize to sum to original total
        if weights.sum() > 0:
            weights = weights / weights.sum() * weights.sum()

        return weights
