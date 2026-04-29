"""
Pure momentum backtest — no value overlay, no crash-risk hedge.
Used as the baseline to compare against the value-hedged strategy.
"""

import os
os.environ['BACKTEST_START'] = '2020-01-01'
os.environ['BACKTEST_END'] = '2025-03-11'

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.parameters import (backtest_config, signal_config,
                               portfolio_config, cost_config, MARKET_TICKERS)
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceTracker
from data.sp500_constituents import get_sp500_universe
from data.price_data import PriceDataManager
from data.market_data import get_vix_data, get_market_index
from signals.momentum import calculate_momentum_1221, rank_momentum
from portfolio.construction import PortfolioConstructor
from portfolio.rebalancing import Rebalancer
from analytics.metrics import generate_performance_table, calculate_crash_period_performance
from utils.helpers import get_month_end_dates
from utils.logging_config import setup_logging


class PureMomentumEngine:
    """Momentum-only engine — no value filter, no hedge."""

    def __init__(self):
        self.price_manager = PriceDataManager()
        self.portfolio_constructor = PortfolioConstructor()
        self.rebalancer = Rebalancer()
        self.performance_tracker = PerformanceTracker()

        self.portfolio_value = backtest_config.INITIAL_CAPITAL
        self.current_weights = pd.Series(dtype=float)

    def run_backtest(self, start_date, end_date):
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("PURE MOMENTUM (no overlay): %s to %s", start_date, end_date)
        logger.info("=" * 60)

        universe = get_sp500_universe(use_cache=True, extend=True)
        logger.info("Universe: %d tickers", len(universe))

        prices = self.price_manager.get_adjusted_prices(universe, start_date, end_date)
        logger.info("Prices: %d days × %d tickers", len(prices), len(prices.columns))

        momentum_scores = calculate_momentum_1221(prices)

        rebalance_dates = get_month_end_dates(start_date, end_date)
        logger.info("Monthly rebalances: %d", len(rebalance_dates))

        for i, rebal_date in enumerate(tqdm(rebalance_dates, desc="Pure Momentum")):
            date_str = rebal_date.strftime('%Y-%m-%d')

            try:
                self._rebalance(date_str, prices, momentum_scores)

                next_date = (rebalance_dates[i + 1]
                             if i < len(rebalance_dates) - 1
                             else pd.Timestamp(end_date))
                self._mark_to_market(rebal_date, next_date, prices)

            except Exception as e:
                logger.error("Error at %s: %s", date_str, e)

        results = pd.DataFrame({
            'portfolio_value': self.performance_tracker.get_value_series(),
            'returns':         self.performance_tracker.get_returns_series(),
        })

        logger.info("Final value: $%,.2f", self.portfolio_value)
        return results

    def _rebalance(self, date, prices, momentum_scores):
        logger = logging.getLogger(__name__)

        if date not in prices.index or date not in momentum_scores.index:
            logger.warning("Missing data for %s, skipping", date)
            return

        # Pure top-N momentum — no value filter, no extreme-score exclusion
        mom_scores = momentum_scores.loc[date].dropna()
        if len(mom_scores) == 0:
            logger.warning("No momentum scores for %s", date)
            return

        selected = mom_scores.nlargest(portfolio_config.LONG_POSITIONS).index.tolist()
        if len(selected) == 0:
            return

        # Equal-weight (market cap unavailable, same as the overlay run)
        target_weights = self.portfolio_constructor.construct_long_portfolio(
            selected_tickers=selected,
            fundamentals=None,
            prices=prices,
            signals=mom_scores,
        )

        # ── NO hedge applied ──────────────────────────────────────────────────

        new_weights, txn_costs = self.rebalancer.execute_rebalance(
            self.current_weights, target_weights, self.portfolio_value
        )
        self.current_weights = new_weights
        self.portfolio_value -= txn_costs

        turnover = self.rebalancer.calculate_turnover(self.current_weights, target_weights)
        self.performance_tracker.record_rebalance(date, turnover, txn_costs, 0.0)
        logger.info("%s: %d stocks, no hedge, costs=$%,.0f",
                    date, len(selected), txn_costs)

    def _mark_to_market(self, start, end, prices):
        for date in prices.loc[start:end].index:
            if len(self.current_weights) == 0:
                continue
            day   = prices.loc[date]
            prev  = prices.shift(1).loc[date]
            rets  = ((day - prev) / prev).reindex(self.current_weights.index).fillna(0)
            port_ret = float((self.current_weights * rets).sum())
            self.portfolio_value *= (1 + port_ret)
            self.performance_tracker.record_daily_value(date.strftime('%Y-%m-%d'),
                                                        self.portfolio_value)
            self.performance_tracker.record_daily_return(date.strftime('%Y-%m-%d'), port_ret)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    setup_logging(log_level='INFO', log_file='results/no_overlay.log')

    start_date = '2020-01-01'
    end_date   = '2025-03-11'

    engine  = PureMomentumEngine()
    results = engine.run_backtest(start_date, end_date)

    os.makedirs('results/backtest_results', exist_ok=True)
    results.to_csv('results/backtest_results/no_overlay_history.csv')

    # Benchmark
    market_data = get_market_index(start_date, end_date)
    if len(market_data) > 0:
        bench_prices = market_data['Close'] if 'Close' in market_data.columns else market_data.iloc[:, 0]
        if isinstance(bench_prices, pd.DataFrame):
            bench_prices = bench_prices.iloc[:, 0]
        bench_returns = bench_prices.pct_change()
        bench_returns = bench_returns[~bench_returns.index.duplicated(keep='first')]
    else:
        bench_returns = None

    perf = generate_performance_table(
        returns=results['returns'],
        values=results['portfolio_value'],
        benchmark_returns=bench_returns,
    )
    perf.to_csv('results/backtest_results/no_overlay_metrics.csv')

    print("\n" + "=" * 60)
    print("PURE MOMENTUM — PERFORMANCE SUMMARY")
    print("=" * 60)
    print(perf.to_string())

    crash = calculate_crash_period_performance(results['returns'])
    if len(crash) > 0:
        print("\nCrash Period Performance:")
        print(crash.to_string(index=False))

    print(f"\nFinal portfolio value: ${engine.portfolio_value:,.2f}")
