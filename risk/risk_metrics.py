"""
Portfolio risk metrics calculation.
"""

import pandas as pd
import numpy as np


def calculate_portfolio_volatility(returns: pd.Series, window: int = 60) -> float:
    """Calculate rolling portfolio volatility (annualized)."""
    if len(returns) < window:
        return np.nan

    vol = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
    return vol


def calculate_beta(portfolio_returns: pd.Series,
                  market_returns: pd.Series) -> float:
    """Calculate portfolio beta to market."""
    # Align series
    common_idx = portfolio_returns.index.intersection(market_returns.index)

    if len(common_idx) < 30:
        return np.nan

    port_ret = portfolio_returns.loc[common_idx]
    mkt_ret = market_returns.loc[common_idx]

    # Calculate beta
    covariance = port_ret.cov(mkt_ret)
    market_variance = mkt_ret.var()

    if market_variance == 0:
        return np.nan

    beta = covariance / market_variance

    return beta


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.

    Args:
        returns: Return series
        confidence: Confidence level (0.95 = 95%)

    Returns:
        VaR (positive number representing potential loss)
    """
    if len(returns) == 0:
        return np.nan

    var = -returns.quantile(1 - confidence)

    return var


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).

    Average of returns worse than VaR.

    Args:
        returns: Return series
        confidence: Confidence level

    Returns:
        CVaR (positive number representing expected loss in tail)
    """
    if len(returns) == 0:
        return np.nan

    var = returns.quantile(1 - confidence)

    # Average of returns worse than VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return np.nan

    cvar = -tail_returns.mean()

    return cvar
