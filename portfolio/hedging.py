"""
Dynamic hedge management.
"""

import pandas as pd
from typing import Dict, Optional
import logging

from config.parameters import risk_config

logger = logging.getLogger(__name__)


class HedgeManager:
    """Manage dynamic crash hedging."""

    def __init__(self, max_hedge_allocation: Optional[float] = None):
        """
        Initialize HedgeManager.

        Args:
            max_hedge_allocation: Maximum hedge allocation (0-1)
        """
        self.max_hedge_allocation = max_hedge_allocation or risk_config.HEDGE_ALLOCATION_MAX
        self.hedge_history = []

    def calculate_hedge_delta(self,
                             risk_score: float,
                             portfolio_beta: float = 1.0) -> float:
        """
        Calculate optimal hedge ratio.

        Delta = risk_score * portfolio_beta * max_hedge

        Args:
            risk_score: Composite risk score
            portfolio_beta: Portfolio beta to market

        Returns:
            Hedge ratio [0, max_hedge]
        """
        delta = risk_score * portfolio_beta * self.max_hedge_allocation

        # Clamp to [0, max_hedge]
        delta = max(0.0, min(delta, self.max_hedge_allocation))

        return delta

    def get_hedge_instruments(self, date: str) -> Dict[str, float]:
        """
        Specify hedge instruments and allocations.

        For backtest simplicity, model as:
        - Cash allocation (zero return)

        In production, could be:
        - Put options on S&P 500
        - Inverse ETF (e.g., SH)

        Args:
            date: Date for hedge

        Returns:
            Dict with {instrument: allocation}
        """
        # Simplified: just cash
        return {'cash': 1.0}

    def track_hedge_performance(self,
                               date: str,
                               hedge_allocation: float,
                               portfolio_return: float,
                               market_return: float):
        """
        Track hedge effectiveness over time.

        Args:
            date: Date
            hedge_allocation: Hedge allocation used
            portfolio_return: Portfolio return
            market_return: Market return
        """
        self.hedge_history.append({
            'date': date,
            'hedge_allocation': hedge_allocation,
            'portfolio_return': portfolio_return,
            'market_return': market_return
        })

    def get_hedge_cost(self) -> float:
        """
        Calculate drag from hedging.

        Returns:
            Average annual cost of hedging
        """
        if not self.hedge_history:
            return 0.0

        # Simplified: cost = average hedge allocation × opportunity cost
        # (assuming cash earns 0, stocks earn positive)
        df = pd.DataFrame(self.hedge_history)
        avg_hedge = df['hedge_allocation'].mean()

        # Rough estimate: 10% equity return × hedge allocation
        cost = avg_hedge * 0.10

        return cost
