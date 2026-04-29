"""
Performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from config.parameters import RISK_FREE_RATE, CRASH_PERIODS
from utils.helpers import annualize_return, annualize_volatility


def calculate_sharpe_ratio(returns: pd.Series,
                          risk_free_rate: float = None,
                          annualization_factor: int = 252) -> float:
    """
    Sharpe Ratio = (Return - Rf) / Volatility (annualized).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Periods per year

    Returns:
        Sharpe ratio
    """
    risk_free_rate = risk_free_rate or RISK_FREE_RATE

    if len(returns) == 0:
        return np.nan

    # Annualize return and volatility
    ann_return = annualize_return(returns, annualization_factor)
    ann_vol = annualize_volatility(returns, annualization_factor)

    if ann_vol == 0:
        return np.nan

    sharpe = (ann_return - risk_free_rate) / ann_vol

    return sharpe


def calculate_sortino_ratio(returns: pd.Series,
                           risk_free_rate: float = None,
                           annualization_factor: int = 252) -> float:
    """
    Sortino Ratio = (Return - Rf) / Downside Deviation.

    Only penalizes downside volatility.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Periods per year

    Returns:
        Sortino ratio
    """
    risk_free_rate = risk_free_rate or RISK_FREE_RATE

    if len(returns) == 0:
        return np.nan

    # Annualize return
    ann_return = annualize_return(returns, annualization_factor)

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.nan

    downside_std = downside_returns.std() * np.sqrt(annualization_factor)

    if downside_std == 0:
        return np.nan

    sortino = (ann_return - risk_free_rate) / downside_std

    return sortino


def calculate_max_drawdown(values: pd.Series) -> Tuple[float, str, str]:
    """
    Calculate maximum drawdown.

    Args:
        values: Series of portfolio values

    Returns:
        (max_drawdown_pct, peak_date, trough_date)
    """
    if len(values) == 0:
        return np.nan, None, None

    # Calculate running maximum
    running_max = values.expanding().max()

    # Calculate drawdown
    drawdown = (values - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    # Find peak and trough dates
    max_dd_idx = drawdown.idxmin()
    peak_idx = values.loc[:max_dd_idx].idxmax()

    peak_date = str(peak_idx.date()) if hasattr(peak_idx, 'date') else str(peak_idx)
    trough_date = str(max_dd_idx.date()) if hasattr(max_dd_idx, 'date') else str(max_dd_idx)

    return max_dd, peak_date, trough_date


def calculate_calmar_ratio(returns: pd.Series, values: pd.Series) -> float:
    """
    Calmar Ratio = Annualized Return / Max Drawdown.

    Args:
        returns: Series of returns
        values: Series of portfolio values

    Returns:
        Calmar ratio
    """
    ann_return = annualize_return(returns, 252)
    max_dd, _, _ = calculate_max_drawdown(values)

    if max_dd == 0 or np.isnan(max_dd):
        return np.nan

    calmar = ann_return / abs(max_dd)

    return calmar


def calculate_tail_risk(returns: pd.Series, percentile: float = 0.05) -> float:
    """
    Calculate tail risk (CVaR - expected return in worst 5% of days).

    Args:
        returns: Series of returns
        percentile: Percentile for tail (default 0.05 = worst 5%)

    Returns:
        Tail risk (positive number = expected loss)
    """
    if len(returns) == 0:
        return np.nan

    threshold = returns.quantile(percentile)
    tail_returns = returns[returns <= threshold]

    if len(tail_returns) == 0:
        return np.nan

    tail_risk = -tail_returns.mean()

    return tail_risk


def calculate_crash_period_performance(returns: pd.Series,
                                      crash_periods: Dict = None) -> pd.DataFrame:
    """
    Calculate performance during known crash periods.

    Args:
        returns: Series of returns
        crash_periods: Dict of {name: (start_date, end_date)}

    Returns:
        DataFrame with crash period returns
    """
    crash_periods = crash_periods or CRASH_PERIODS

    results = []

    for period_name, (start, end) in crash_periods.items():
        try:
            period_returns = returns.loc[start:end]

            if len(period_returns) > 0:
                total_return = (1 + period_returns).prod() - 1

                results.append({
                    'period': period_name,
                    'start_date': start,
                    'end_date': end,
                    'return': total_return,
                    'num_days': len(period_returns)
                })
        except:
            pass

    return pd.DataFrame(results)


def calculate_information_ratio(portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series) -> float:
    """
    Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    # Remove duplicates
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]
    benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep='first')]

    # Align returns
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)

    if len(common_idx) < 30:
        return np.nan

    port_ret = portfolio_returns.loc[common_idx]
    bench_ret = benchmark_returns.loc[common_idx]

    # Excess returns
    excess = port_ret - bench_ret

    # Annualize
    ann_excess = float(annualize_return(excess, 252))
    tracking_error = float(annualize_volatility(excess, 252))

    if tracking_error == 0:
        return np.nan

    info_ratio = ann_excess / tracking_error

    return info_ratio


def generate_performance_table(returns: pd.Series,
                              values: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Comprehensive performance table.

    Args:
        returns: Portfolio returns
        values: Portfolio values
        benchmark_returns: Benchmark returns (optional)

    Returns:
        DataFrame with performance metrics
    """
    metrics = {}

    # Basic metrics
    metrics['Total Return'] = (values.iloc[-1] / values.iloc[0] - 1) if len(values) > 0 else np.nan
    metrics['Annualized Return'] = annualize_return(returns, 252)
    metrics['Volatility'] = annualize_volatility(returns, 252)

    # Risk-adjusted metrics
    metrics['Sharpe Ratio'] = calculate_sharpe_ratio(returns)
    metrics['Sortino Ratio'] = calculate_sortino_ratio(returns)

    # Drawdown metrics
    max_dd, peak_date, trough_date = calculate_max_drawdown(values)
    metrics['Max Drawdown'] = max_dd
    metrics['Calmar Ratio'] = calculate_calmar_ratio(returns, values)

    # Tail risk
    metrics['Tail Risk (CVaR 5%)'] = calculate_tail_risk(returns)

    # Win rate
    metrics['Win Rate'] = (returns > 0).sum() / len(returns) if len(returns) > 0 else np.nan

    # Benchmark comparison
    if benchmark_returns is not None:
        metrics['Information Ratio'] = calculate_information_ratio(returns, benchmark_returns)

    # Convert to DataFrame
    df = pd.DataFrame([metrics]).T
    df.columns = ['Value']

    return df
