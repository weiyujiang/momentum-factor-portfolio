"""
Value-Hedged Momentum Portfolio Backtest
Main execution script
"""

import logging
import os
from config.parameters import backtest_config, MARKET_TICKERS
from backtest.engine import BacktestEngine
from analytics.metrics import generate_performance_table, calculate_crash_period_performance
from analytics.visualization import plot_cumulative_returns, plot_drawdown, plot_rolling_sharpe
from data.market_data import get_market_index
from utils.logging_config import setup_logging
import pandas as pd


def main():
    """Main execution function."""

    # Setup logging
    setup_logging(log_level='INFO', log_file='results/backtest.log')
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("VALUE-HEDGED MOMENTUM PORTFOLIO BACKTEST")
    logger.info("="*70)

    # Configuration
    start_date = backtest_config.START_DATE
    end_date = backtest_config.END_DATE

    logger.info(f"Backtest Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${backtest_config.INITIAL_CAPITAL:,.0f}")

    # Initialize backtest engine
    logger.info("\nInitializing backtest engine...")
    engine = BacktestEngine()

    # Run backtest
    logger.info("\nRunning backtest...\n")
    results = engine.run_backtest(start_date=start_date, end_date=end_date)

    # Save results
    logger.info("\nSaving backtest results...")
    os.makedirs('results/backtest_results', exist_ok=True)
    results.to_csv('results/backtest_results/portfolio_history.csv')
    logger.info("Saved: results/backtest_results/portfolio_history.csv")

    # Generate analytics
    logger.info("\nGenerating performance analytics...")

    # Load benchmark (S&P 500)
    logger.info("Loading S&P 500 benchmark...")
    try:
        market_data = get_market_index(start=start_date, end=end_date)
        if len(market_data) > 0:
            benchmark_prices = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
            benchmark_returns = benchmark_prices.pct_change()
            # Remove any duplicate index values
            benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep='first')]
        else:
            logger.warning("Could not load benchmark data")
            benchmark_returns = None
    except Exception as e:
        logger.error(f"Error loading benchmark: {e}")
        benchmark_returns = None

    # Performance table
    logger.info("Calculating performance metrics...")
    perf_table = generate_performance_table(
        returns=results['returns'],
        values=results['portfolio_value'],
        benchmark_returns=benchmark_returns
    )

    perf_table.to_csv('results/backtest_results/performance_metrics.csv')
    logger.info("Saved: results/backtest_results/performance_metrics.csv")

    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*70)
    print(perf_table.to_string())

    # Crash period analysis
    logger.info("\nAnalyzing crash period performance...")
    crash_perf = calculate_crash_period_performance(results['returns'])
    if len(crash_perf) > 0:
        crash_perf.to_csv('results/backtest_results/crash_period_analysis.csv', index=False)
        logger.info("Saved: results/backtest_results/crash_period_analysis.csv")
        print("\nCrash Period Performance:")
        print(crash_perf.to_string(index=False))

    # Visualizations
    logger.info("\nGenerating visualizations...")
    os.makedirs('results/figures', exist_ok=True)

    try:
        # Cumulative returns
        logger.info("Creating cumulative returns plot...")
        benchmark_values = benchmark_prices if 'benchmark_prices' in locals() else None
        plot_cumulative_returns(
            portfolio_values=results['portfolio_value'],
            benchmark_values=benchmark_values,
            save_path='results/figures/cumulative_returns.png'
        )
        logger.info("Saved: results/figures/cumulative_returns.png")

        # Drawdown
        logger.info("Creating drawdown plot...")
        plot_drawdown(
            values=results['portfolio_value'],
            save_path='results/figures/drawdown.png'
        )
        logger.info("Saved: results/figures/drawdown.png")

        # Rolling Sharpe
        logger.info("Creating rolling Sharpe plot...")
        plot_rolling_sharpe(
            returns=results['returns'],
            window=252,
            save_path='results/figures/rolling_sharpe.png'
        )
        logger.info("Saved: results/figures/rolling_sharpe.png")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

    logger.info("\n" + "="*70)
    logger.info("BACKTEST COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: results/")
    logger.info("")

    return results, perf_table


if __name__ == '__main__':
    results, summary = main()
