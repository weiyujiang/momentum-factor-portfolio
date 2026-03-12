"""
Portfolio rebalancing with transaction costs.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from config.parameters import cost_config

logger = logging.getLogger(__name__)


class Rebalancer:
    """Portfolio rebalancing logic."""

    def __init__(self,
                 commission: Optional[float] = None,
                 slippage: Optional[float] = None,
                 market_impact: Optional[float] = None):
        """
        Initialize Rebalancer.

        Args:
            commission: Commission per trade (bps)
            slippage: Slippage per trade (bps)
            market_impact: Market impact per trade (bps)
        """
        self.commission = commission or cost_config.COMMISSION
        self.slippage = slippage or cost_config.SLIPPAGE
        self.market_impact = market_impact or cost_config.MARKET_IMPACT

    def calculate_trades(self,
                        current_weights: pd.Series,
                        target_weights: pd.Series,
                        portfolio_value: float) -> pd.DataFrame:
        """
        Calculate required trades to reach target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Current portfolio value ($)

        Returns:
            DataFrame with columns: [ticker, current_weight, target_weight,
                                    weight_change, trade_value]
        """
        # Combine current and target
        all_tickers = current_weights.index.union(target_weights.index)

        trades = []
        for ticker in all_tickers:
            current_w = current_weights.get(ticker, 0.0)
            target_w = target_weights.get(ticker, 0.0)

            weight_change = target_w - current_w
            trade_value = weight_change * portfolio_value

            trades.append({
                'ticker': ticker,
                'current_weight': current_w,
                'target_weight': target_w,
                'weight_change': weight_change,
                'trade_value': trade_value
            })

        trades_df = pd.DataFrame(trades)

        return trades_df

    def calculate_transaction_costs(self,
                                   trades: pd.DataFrame) -> float:
        """
        Calculate total transaction costs.

        Total Cost = (Commission + Slippage + Market Impact) × |Trade Value|

        Args:
            trades: DataFrame with trade information

        Returns:
            Total cost in dollars
        """
        # Total cost per trade (bps)
        cost_per_trade = self.commission + self.slippage + self.market_impact

        # Apply to absolute trade values
        abs_trade_values = trades['trade_value'].abs()
        total_cost = (abs_trade_values * cost_per_trade).sum()

        return total_cost

    def execute_rebalance(self,
                         current_weights: pd.Series,
                         target_weights: pd.Series,
                         portfolio_value: float) -> Tuple[pd.Series, float]:
        """
        Execute rebalancing and return new weights and costs.

        Args:
            current_weights: Current weights
            target_weights: Target weights
            portfolio_value: Portfolio value

        Returns:
            (new_weights, transaction_costs)
        """
        # Calculate trades
        trades = self.calculate_trades(current_weights, target_weights, portfolio_value)

        # Calculate costs
        costs = self.calculate_transaction_costs(trades)

        # New weights are just target weights (we execute fully)
        new_weights = target_weights.copy()

        logger.info(f"Rebalance: {len(trades)} trades, ${costs:,.2f} in costs")

        return new_weights, costs

    def calculate_turnover(self,
                          current_weights: pd.Series,
                          target_weights: pd.Series) -> float:
        """
        Calculate portfolio turnover.

        Turnover = sum(|weight_changes|) / 2

        Args:
            current_weights: Current weights
            target_weights: Target weights

        Returns:
            Turnover (0-1)
        """
        all_tickers = current_weights.index.union(target_weights.index)

        weight_changes = []
        for ticker in all_tickers:
            current_w = current_weights.get(ticker, 0.0)
            target_w = target_weights.get(ticker, 0.0)
            weight_changes.append(abs(target_w - current_w))

        turnover = sum(weight_changes) / 2

        return turnover
