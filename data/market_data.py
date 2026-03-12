"""
Market-wide data for crash risk indicators (VIX, S&P 500, Treasury rates).
"""

import pandas as pd
import yfinance as yf
import logging
import os

from config.parameters import MARKET_TICKERS, cache_config
from utils.cache import save_to_pickle, load_from_pickle, is_cache_valid

logger = logging.getLogger(__name__)


def get_vix_data(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download VIX data from yfinance.

    Ticker: ^VIX

    Args:
        start: Start date ('YYYY-MM-DD')
        end: End date ('YYYY-MM-DD')
        use_cache: Whether to use cached data

    Returns:
        DataFrame with date index and VIX close prices
    """
    logger.info(f"Downloading VIX data from {start} to {end}")

    cache_file = os.path.join(cache_config.CACHE_DIR, 'vix_data.pkl')

    # Try cache first
    if use_cache and is_cache_valid(cache_file, max_age_days=cache_config.CACHE_MAX_AGE_DAYS):
        logger.info("Loading VIX data from cache")
        vix_data = load_from_pickle(cache_file)
        if vix_data is not None:
            # Filter to requested date range
            return vix_data.loc[start:end]

    # Download from yfinance
    try:
        vix = yf.download(MARKET_TICKERS['vix'], start=start, end=end, progress=False)

        if len(vix) > 0:
            vix_data = pd.DataFrame({
                'VIX': vix['Adj Close'] if 'Adj Close' in vix.columns else vix['Close']
            })

            # Save to cache
            save_to_pickle(vix_data, cache_file)

            logger.info(f"Downloaded {len(vix_data)} days of VIX data")
            return vix_data

    except Exception as e:
        logger.error(f"Error downloading VIX data: {e}")

    return pd.DataFrame()


def get_market_index(start: str, end: str,
                    ticker: str = None,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    Download market index data (S&P 500).

    Args:
        start: Start date
        end: End date
        ticker: Market index ticker (default: ^GSPC for S&P 500)
        use_cache: Whether to use cached data

    Returns:
        DataFrame with date index and price/volume data
    """
    if ticker is None:
        ticker = MARKET_TICKERS['sp500']

    logger.info(f"Downloading {ticker} data from {start} to {end}")

    cache_file = os.path.join(cache_config.CACHE_DIR, f'{ticker.replace("^", "")}_data.pkl')

    # Try cache first
    if use_cache and is_cache_valid(cache_file, max_age_days=cache_config.CACHE_MAX_AGE_DAYS):
        logger.info(f"Loading {ticker} data from cache")
        data = load_from_pickle(cache_file)
        if data is not None:
            return data.loc[start:end]

    # Download from yfinance
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)

        if len(data) > 0:
            # Save to cache
            save_to_pickle(data, cache_file)

            logger.info(f"Downloaded {len(data)} days of {ticker} data")
            return data

    except Exception as e:
        logger.error(f"Error downloading {ticker} data: {e}")

    return pd.DataFrame()


def get_treasury_rates(start: str, end: str,
                      ticker: str = None,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Download Treasury yield data.

    Args:
        start: Start date
        end: End date
        ticker: Treasury ticker (default: ^TNX for 10-year)
        use_cache: Whether to use cached data

    Returns:
        DataFrame with date index and yield data
    """
    if ticker is None:
        ticker = MARKET_TICKERS['treasury_10y']

    logger.info(f"Downloading {ticker} data from {start} to {end}")

    cache_file = os.path.join(cache_config.CACHE_DIR, f'{ticker.replace("^", "")}_data.pkl')

    # Try cache first
    if use_cache and is_cache_valid(cache_file, max_age_days=cache_config.CACHE_MAX_AGE_DAYS):
        logger.info(f"Loading {ticker} data from cache")
        data = load_from_pickle(cache_file)
        if data is not None:
            return data.loc[start:end]

    # Download from yfinance
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)

        if len(data) > 0:
            # Save to cache
            save_to_pickle(data, cache_file)

            logger.info(f"Downloaded {len(data)} days of {ticker} data")
            return data

    except Exception as e:
        logger.error(f"Error downloading {ticker} data: {e}")

    return pd.DataFrame()


def get_risk_free_rate(start: str, end: str) -> pd.Series:
    """
    Get risk-free rate time series (10-year Treasury yield).

    Args:
        start: Start date
        end: End date

    Returns:
        Series with risk-free rates (annualized)
    """
    treasury_data = get_treasury_rates(start, end)

    if len(treasury_data) > 0 and 'Close' in treasury_data.columns:
        # Treasury yields from yfinance are in percentage points
        # Convert to decimal (e.g., 2.5% -> 0.025)
        rf_rate = treasury_data['Close'] / 100
        return rf_rate

    # Fallback to constant rate
    logger.warning("Could not download Treasury data, using constant risk-free rate")
    from config.parameters import RISK_FREE_RATE
    dates = pd.date_range(start=start, end=end, freq='D')
    return pd.Series(RISK_FREE_RATE, index=dates)


def calculate_market_returns(start: str, end: str) -> pd.Series:
    """
    Calculate market returns (S&P 500).

    Args:
        start: Start date
        end: End date

    Returns:
        Series with daily market returns
    """
    market_data = get_market_index(start, end)

    if len(market_data) > 0:
        prices = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
        returns = prices.pct_change()
        return returns

    return pd.Series()
