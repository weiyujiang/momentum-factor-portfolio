"""
Crash risk indicators for dynamic hedging.

Monitors: VIX, momentum dispersion, value spread.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from config.parameters import risk_config
from signals.momentum import calculate_momentum_dispersion
from signals.value import calculate_value_spread

logger = logging.getLogger(__name__)


class CrashRiskMonitor:
    """Monitor multiple crash risk indicators."""

    def __init__(self,
                 vix_threshold: Optional[float] = None,
                 dispersion_threshold: Optional[float] = None,
                 spread_threshold: Optional[float] = None):
        """
        Initialize CrashRiskMonitor.

        Args:
            vix_threshold: VIX level indicating elevated risk
            dispersion_threshold: Momentum dispersion threshold
            spread_threshold: Value spread threshold
        """
        self.vix_threshold = vix_threshold or risk_config.VIX_THRESHOLD
        self.dispersion_threshold = dispersion_threshold or risk_config.DISPERSION_THRESHOLD
        self.spread_threshold = spread_threshold or risk_config.VALUE_SPREAD_THRESHOLD

    def calculate_vix_signal(self, vix_level: float) -> float:
        """
        VIX-based risk signal.

        Signal = 0 if VIX < threshold
        Signal = (VIX - threshold) / threshold if VIX >= threshold

        Args:
            vix_level: Current VIX level

        Returns:
            Risk signal [0, inf)
        """
        if vix_level < self.vix_threshold:
            return 0.0

        signal = (vix_level - self.vix_threshold) / self.vix_threshold

        return signal

    def calculate_dispersion_signal(self,
                                    momentum_scores: pd.DataFrame,
                                    date: str) -> float:
        """
        Momentum dispersion signal.

        High cross-sectional dispersion suggests regime instability.

        Args:
            momentum_scores: DataFrame with momentum scores
            date: Date for calculation

        Returns:
            Risk signal [0, inf)
        """
        dispersion = calculate_momentum_dispersion(momentum_scores, date)

        if pd.isna(dispersion) or dispersion < self.dispersion_threshold:
            return 0.0

        signal = (dispersion - self.dispersion_threshold) / self.dispersion_threshold

        return signal

    def calculate_spread_signal(self,
                               fundamentals: pd.DataFrame) -> float:
        """
        Value spread signal.

        High spread (expensive stocks very expensive) suggests bubble.

        Args:
            fundamentals: DataFrame with fundamental data

        Returns:
            Risk signal [0, inf)
        """
        spread = calculate_value_spread(fundamentals)

        if pd.isna(spread) or spread < self.spread_threshold:
            return 0.0

        signal = (spread - self.spread_threshold) / self.spread_threshold

        return signal

    def get_composite_risk(self,
                          vix: float,
                          momentum_scores: Optional[pd.DataFrame] = None,
                          fundamentals: Optional[pd.DataFrame] = None,
                          date: Optional[str] = None,
                          weights: Optional[List[float]] = None) -> float:
        """
        Calculate composite crash risk.

        Composite = w1*VIX_signal + w2*Dispersion_signal + w3*Spread_signal

        Args:
            vix: Current VIX level
            momentum_scores: Momentum scores DataFrame
            fundamentals: Fundamentals DataFrame
            date: Date for calculation
            weights: Custom weights [w_vix, w_dispersion, w_spread]

        Returns:
            Composite risk score [0, inf)
        """
        if weights is None:
            weights = [
                risk_config.VIX_WEIGHT,
                risk_config.DISPERSION_WEIGHT,
                risk_config.SPREAD_WEIGHT
            ]

        # Calculate individual signals
        vix_signal = self.calculate_vix_signal(vix)

        dispersion_signal = 0.0
        if momentum_scores is not None and date is not None:
            dispersion_signal = self.calculate_dispersion_signal(momentum_scores, date)

        spread_signal = 0.0
        if fundamentals is not None:
            spread_signal = self.calculate_spread_signal(fundamentals)

        # Composite risk
        composite = (
            weights[0] * vix_signal +
            weights[1] * dispersion_signal +
            weights[2] * spread_signal
        )

        return composite

    def get_hedge_allocation(self,
                           risk_score: float,
                           max_hedge: Optional[float] = None) -> float:
        """
        Determine hedge allocation based on risk score.

        Linear scaling: hedge = min(risk_score * max_hedge, max_hedge)

        Args:
            risk_score: Composite risk score
            max_hedge: Maximum hedge allocation

        Returns:
            Hedge allocation [0, max_hedge]
        """
        max_hedge = max_hedge or risk_config.HEDGE_ALLOCATION_MAX

        # Linear scaling with cap
        hedge = min(risk_score * max_hedge, max_hedge)

        # Floor at 0
        hedge = max(hedge, 0.0)

        return hedge
