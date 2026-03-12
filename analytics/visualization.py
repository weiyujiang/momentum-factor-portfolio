"""
Visualization functions for backtest results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_cumulative_returns(portfolio_values: pd.Series,
                           benchmark_values: Optional[pd.Series] = None,
                           save_path: Optional[str] = None):
    """
    Plot cumulative returns vs benchmark.

    Args:
        portfolio_values: Portfolio value series
        benchmark_values: Benchmark value series
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize to 100
    port_norm = portfolio_values / portfolio_values.iloc[0] * 100
    ax.plot(port_norm.index, port_norm.values, label='Value-Hedged Momentum', linewidth=2)

    if benchmark_values is not None:
        bench_norm = benchmark_values / benchmark_values.iloc[0] * 100
        ax.plot(bench_norm.index, bench_norm.values, label='S&P 500', linewidth=2, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Value (Base = 100)')
    ax.set_title('Cumulative Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_drawdown(values: pd.Series, save_path: Optional[str] = None):
    """
    Plot drawdown over time.

    Args:
        values: Portfolio value series
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate drawdown
    running_max = values.expanding().max()
    drawdown = (values - running_max) / running_max

    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)

    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title('Portfolio Drawdown')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rolling_sharpe(returns: pd.Series,
                       window: int = 252,
                       save_path: Optional[str] = None):
    """
    Plot rolling Sharpe ratio.

    Args:
        returns: Return series
        window: Rolling window size
        save_path: Path to save figure
    """
    from utils.helpers import annualize_return, annualize_volatility
    from config.parameters import RISK_FREE_RATE

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate rolling Sharpe
    rolling_mean = returns.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - RISK_FREE_RATE) / rolling_std

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5, label='Sharpe = 1.0')

    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Sharpe Ratio')
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
