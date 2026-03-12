"""
Main backtesting engine orchestrating all components.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from tqdm import tqdm

from config.parameters import (backtest_config, signal_config, portfolio_config,
                               risk_config, cost_config)
from data.sp500_constituents import get_sp500_universe, get_universe_at_date
from data.price_data import PriceDataManager
from data.fundamental_data import FundamentalDataManager
from data.market_data import get_vix_data, get_market_index
from signals.momentum import calculate_momentum_1221, rank_momentum
from signals.value import calculate_value_composite
from signals.combined import generate_combined_signal, select_top_stocks
from risk.crash_indicators import CrashRiskMonitor
from portfolio.construction import PortfolioConstructor
from portfolio.rebalancing import Rebalancer
from portfolio.hedging import HedgeManager
from backtest.performance import PerformanceTracker
from utils.helpers import get_month_end_dates

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize BacktestEngine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize managers
        self.price_manager = PriceDataManager()
        self.fundamental_manager = FundamentalDataManager()
        self.portfolio_constructor = PortfolioConstructor()
        self.rebalancer = Rebalancer()
        self.crash_monitor = CrashRiskMonitor()
        self.hedge_manager = HedgeManager()
        self.performance_tracker = PerformanceTracker()

        # State tracking
        self.portfolio_value = backtest_config.INITIAL_CAPITAL
        self.current_weights = pd.Series()
        self.current_positions = {}

    def run_backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Main backtest loop.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            DataFrame with backtest results
        """
        logger.info(f"="*60)
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        logger.info(f"="*60)

        # Step 1: Get universe
        logger.info("Loading universe...")
        universe = get_sp500_universe(use_cache=True, extend=True)
        logger.info(f"Universe size: {len(universe)} tickers")

        # Step 2: Download data
        logger.info("Downloading price data...")
        prices = self.price_manager.get_adjusted_prices(universe, start_date, end_date)
        logger.info(f"Downloaded prices: {len(prices)} days, {len(prices.columns)} tickers")

        # Download VIX data
        logger.info("Downloading VIX data...")
        vix_data = get_vix_data(start_date, end_date)

        # Download market data
        logger.info("Downloading S&P 500 benchmark...")
        market_data = get_market_index(start_date, end_date)

        # Step 3: Calculate momentum signals
        logger.info("Calculating momentum signals...")
        momentum_scores = calculate_momentum_1221(prices)

        # Step 4: Get rebalance dates
        rebalance_dates = get_month_end_dates(start_date, end_date)
        logger.info(f"Rebalancing monthly: {len(rebalance_dates)} periods")

        # Step 5: Run backtest loop
        logger.info("Running backtest...")

        for i, rebal_date in enumerate(tqdm(rebalance_dates, desc="Backtest Progress")):
            rebal_date_str = rebal_date.strftime('%Y-%m-%d')

            try:
                # Rebalance portfolio
                self._rebalance_portfolio(
                    rebal_date_str,
                    prices,
                    momentum_scores,
                    vix_data
                )

                # Mark to market daily until next rebalance
                if i < len(rebalance_dates) - 1:
                    next_rebal = rebalance_dates[i + 1]
                else:
                    next_rebal = pd.Timestamp(end_date)

                self._mark_to_market_period(
                    rebal_date,
                    next_rebal,
                    prices
                )

            except Exception as e:
                logger.error(f"Error at {rebal_date_str}: {e}")
                continue

        # Step 6: Compile results
        logger.info("Compiling results...")
        results = self._compile_results()

        logger.info("="*60)
        logger.info("Backtest complete!")
        logger.info(f"Final portfolio value: ${self.portfolio_value:,.2f}")
        logger.info("="*60)

        return results

    def _rebalance_portfolio(self,
                            date: str,
                            prices: pd.DataFrame,
                            momentum_scores: pd.DataFrame,
                            vix_data: pd.DataFrame):
        """Execute single rebalancing step."""

        # Get data for this date
        if date not in prices.index or date not in momentum_scores.index:
            logger.warning(f"Missing data for {date}, skipping rebalance")
            return

        # Get momentum scores for this date
        mom_scores = momentum_scores.loc[date].dropna()

        # Get fundamentals (simplified - use most recent available)
        # In production, would use point-in-time fundamentals
        available_tickers = mom_scores.index.tolist()

        # For simplicity, skip fundamental download during each rebalance
        # Instead, use a simplified value proxy or pre-cached fundamentals
        # Here we'll just use momentum with a value tilt approximation

        # Select top stocks by momentum
        top_momentum = rank_momentum(momentum_scores, date, top_n=portfolio_config.LONG_POSITIONS * 2)

        # Simple value filter: exclude stocks with extreme momentum (likely overpriced)
        # This is a simplified version - in production, use actual fundamentals
        value_filtered = top_momentum[top_momentum < top_momentum.quantile(0.9)]

        # Select final portfolio
        selected_tickers = value_filtered.head(portfolio_config.LONG_POSITIONS).index.tolist()

        if len(selected_tickers) == 0:
            logger.warning(f"No stocks selected for {date}")
            return

        # Get VIX level for crash risk
        vix_level = 25.0  # Default
        if date in vix_data.index and 'VIX' in vix_data.columns:
            vix_level = vix_data.loc[date, 'VIX']

        # Calculate crash risk
        risk_score = self.crash_monitor.get_composite_risk(
            vix=vix_level,
            momentum_scores=momentum_scores,
            date=date
        )

        # Determine hedge allocation
        hedge_allocation = self.crash_monitor.get_hedge_allocation(risk_score)

        # Construct target portfolio
        target_weights = self.portfolio_constructor.construct_long_portfolio(
            selected_tickers=selected_tickers,
            fundamentals=None,  # Simplified
            prices=prices,
            signals=mom_scores
        )

        # Apply hedge
        target_weights = self.portfolio_constructor.apply_hedge(
            target_weights,
            hedge_allocation
        )

        # Execute rebalancing
        new_weights, txn_costs = self.rebalancer.execute_rebalance(
            self.current_weights,
            target_weights,
            self.portfolio_value
        )

        # Update state
        self.current_weights = new_weights
        self.portfolio_value -= txn_costs  # Deduct transaction costs

        # Track performance
        turnover = self.rebalancer.calculate_turnover(self.current_weights, target_weights)
        self.performance_tracker.record_rebalance(date, turnover, txn_costs, hedge_allocation)

        logger.info(f"{date}: Selected {len(selected_tickers)} stocks, VIX={vix_level:.1f}, "
                   f"Risk={risk_score:.2f}, Hedge={hedge_allocation:.1%}, Costs=${txn_costs:,.0f}")

    def _mark_to_market_period(self,
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp,
                              prices: pd.DataFrame):
        """Mark to market for period between rebalances."""

        # Get daily dates in period
        daily_dates = prices.loc[start_date:end_date].index

        for date in daily_dates:
            date_str = date.strftime('%Y-%m-%d')

            # Calculate portfolio return
            if len(self.current_weights) > 0:
                # Get returns for holdings
                day_prices = prices.loc[date]
                prev_prices = prices.shift(1).loc[date]

                returns = (day_prices - prev_prices) / prev_prices
                returns = returns[self.current_weights.index].fillna(0)

                # Portfolio return
                port_return = (self.current_weights * returns).sum()

                # Update value
                self.portfolio_value *= (1 + port_return)

                # Track
                self.performance_tracker.record_daily_value(date_str, self.portfolio_value)
                self.performance_tracker.record_daily_return(date_str, port_return)

    def _compile_results(self) -> pd.DataFrame:
        """Compile backtest results into DataFrame."""

        values = self.performance_tracker.get_value_series()
        returns = self.performance_tracker.get_returns_series()

        results = pd.DataFrame({
            'portfolio_value': values,
            'returns': returns
        })

        return results

    def get_performance_summary(self) -> Dict:
        """Generate performance summary statistics."""
        return self.performance_tracker.get_performance_summary()
