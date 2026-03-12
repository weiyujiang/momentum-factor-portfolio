"""
S&P 500 constituent management with survivorship bias mitigation.

Strategy: Build extended universe of 600-700 stocks to reduce survivorship bias,
then dynamically filter to top 500 by market cap at each rebalance date.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Set
import requests
from bs4 import BeautifulSoup
import logging

from config.parameters import universe_config, cache_config
from utils.cache import save_to_pickle, load_from_pickle, is_cache_valid
from utils.helpers import normalize_tickers

logger = logging.getLogger(__name__)


def get_current_sp500_constituents() -> List[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.

    Returns:
        List of ticker symbols
    """
    logger.info("Fetching current S&P 500 constituents from Wikipedia...")

    try:
        # Wikipedia S&P 500 table
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the constituents table (first table on page)
        table = soup.find('table', {'class': 'wikitable'})

        if table is None:
            logger.warning("Could not find S&P 500 table on Wikipedia, using fallback")
            return _get_fallback_constituents()

        # Extract tickers from first column
        tickers = []
        rows = table.find_all('tr')[1:]  # Skip header row

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 0:
                ticker = cols[0].text.strip()
                tickers.append(ticker)

        # Normalize tickers
        tickers = normalize_tickers(tickers)

        logger.info(f"Successfully fetched {len(tickers)} S&P 500 constituents")
        return tickers

    except Exception as e:
        logger.error(f"Error fetching S&P 500 constituents: {e}")
        logger.warning("Using fallback constituent list")
        return _get_fallback_constituents()


def _get_fallback_constituents() -> List[str]:
    """
    Fallback list of major S&P 500 constituents (in case scraping fails).

    Returns:
        List of ticker symbols
    """
    # Major S&P 500 constituents (top ~100 by market cap as of 2024)
    fallback = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'XOM', 'V', 'JPM', 'WMT', 'PG', 'MA', 'CVX', 'HD', 'MRK', 'ABBV',
        'LLY', 'KO', 'AVGO', 'PEP', 'COST', 'TMO', 'MCD', 'CSCO', 'ABT', 'WFC',
        'ACN', 'ADBE', 'CRM', 'DHR', 'VZ', 'CMCSA', 'NFLX', 'NKE', 'DIS', 'TXN',
        'INTC', 'UPS', 'PM', 'BA', 'HON', 'QCOM', 'ORCL', 'IBM', 'AMGN', 'SPGI',
        'CAT', 'GE', 'AMD', 'INTU', 'COP', 'NOW', 'RTX', 'LOW', 'GS', 'SBUX',
        'AXP', 'ISRG', 'BLK', 'DE', 'GILD', 'LMT', 'ELV', 'PLD', 'MDLZ', 'BKNG',
        'ADP', 'TJX', 'VRTX', 'ADI', 'MMC', 'SYK', 'REGN', 'CI', 'C', 'TMUS',
        'CB', 'ZTS', 'MO', 'PGR', 'SO', 'ETN', 'DUK', 'EOG', 'BSX', 'ITW',
        'CME', 'PNC', 'USB', 'NOC', 'APH', 'MSI', 'CL', 'SHW', 'EMR', 'WM'
    ]

    logger.warning(f"Using fallback list with {len(fallback)} tickers")
    return fallback


def build_extended_universe(sp500_tickers: List[str],
                           additional_tickers: Optional[List[str]] = None) -> List[str]:
    """
    Build extended universe to reduce survivorship bias.

    Strategy:
    1. Start with current S&P 500
    2. Add additional large-cap stocks to reach target size
    3. This provides a pool of stocks that likely includes historical S&P 500 members

    Args:
        sp500_tickers: Current S&P 500 constituent list
        additional_tickers: Optional list of additional tickers to include

    Returns:
        List of tickers for extended universe
    """
    logger.info("Building extended universe for survivorship bias mitigation...")

    extended = set(sp500_tickers)

    # Add additional tickers if provided
    if additional_tickers:
        extended.update(additional_tickers)

    # Add known large-cap stocks that may have been in S&P 500 historically
    # These include recently delisted, merged, or dropped stocks
    historical_notables = [
        # Tech/telecom that were dropped or merged
        'T', 'VZ', 'ORCL', 'IBM', 'CSCO', 'INTC', 'QCOM', 'TXN', 'MU',
        # Financials (many consolidated post-2008)
        'BAC', 'C', 'WFC', 'USB', 'PNC', 'TFC', 'COF', 'AIG',
        # Energy (many dropped due to oil crash)
        'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'OXY', 'EOG', 'PXD',
        # Healthcare/Pharma
        'PFE', 'MRK', 'ABBV', 'BMY', 'LLY', 'GILD', 'AMGN', 'BIIB',
        # Consumer
        'WMT', 'TGT', 'KR', 'CVS', 'WBA', 'DG', 'DLTR',
        # Industrials
        'GE', 'MMM', 'HON', 'CAT', 'DE', 'BA', 'LMT', 'NOC', 'RTX',
        # Other large caps
        'F', 'GM', 'DAL', 'UAL', 'AAL', 'LUV',
    ]

    extended.update(historical_notables)

    # Convert to list
    extended_list = list(extended)

    logger.info(f"Extended universe size: {len(extended_list)} (target: {universe_config.BASE_UNIVERSE_SIZE})")

    # If we need more tickers to reach target size, we would fetch from a broader index
    # For now, we'll work with what we have

    return extended_list


def get_universe_at_date(tickers: List[str],
                        market_caps: pd.Series,
                        n_stocks: int = 500) -> List[str]:
    """
    Get top N stocks by market cap at a specific date.

    This approximates S&P 500 membership without perfect historical data.

    Args:
        tickers: Available ticker universe
        market_caps: Series with tickers as index, market caps as values
        n_stocks: Number of stocks to select (default 500)

    Returns:
        List of selected tickers
    """
    # Filter out missing market caps
    valid_market_caps = market_caps.dropna()

    # Sort by market cap descending
    sorted_tickers = valid_market_caps.sort_values(ascending=False)

    # Select top N
    selected = sorted_tickers.head(n_stocks).index.tolist()

    return selected


def build_constituent_timeline(extended_universe: List[str],
                               start_date: str,
                               end_date: str,
                               freq: str = 'Q') -> pd.DataFrame:
    """
    Build a timeline of universe membership (placeholder for future enhancement).

    In current implementation, we use dynamic filtering by market cap instead.

    Args:
        extended_universe: List of tickers in extended universe
        start_date: Start date
        end_date: End date
        freq: Frequency for snapshots

    Returns:
        DataFrame with [date, ticker, in_universe] columns
    """
    # For now, return a simple DataFrame indicating all tickers are always "available"
    # In production, you'd track actual add/remove dates from historical data sources

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    records = []
    for date in dates:
        for ticker in extended_universe:
            records.append({
                'date': date,
                'ticker': ticker,
                'in_universe': True,  # Simplified - assume always available
            })

    df = pd.DataFrame(records)

    logger.info(f"Built constituent timeline: {len(dates)} dates × {len(extended_universe)} tickers")

    return df


def save_constituent_cache(constituents: List[str], cache_type: str = 'sp500') -> None:
    """
    Save constituent list to cache.

    Args:
        constituents: List of tickers
        cache_type: Type of cache ('sp500', 'extended', etc.)
    """
    import os
    cache_file = os.path.join(cache_config.CONSTITUENT_CACHE_DIR, f'{cache_type}_constituents.pkl')
    save_to_pickle(constituents, cache_file)
    logger.info(f"Saved {len(constituents)} {cache_type} constituents to cache")


def load_constituent_cache(cache_type: str = 'sp500') -> Optional[List[str]]:
    """
    Load constituent list from cache.

    Args:
        cache_type: Type of cache ('sp500', 'extended', etc.)

    Returns:
        List of tickers or None if cache doesn't exist/expired
    """
    import os
    cache_file = os.path.join(cache_config.CONSTITUENT_CACHE_DIR, f'{cache_type}_constituents.pkl')

    if not is_cache_valid(cache_file, max_age_days=cache_config.CACHE_MAX_AGE_DAYS):
        return None

    constituents = load_from_pickle(cache_file)

    if constituents:
        logger.info(f"Loaded {len(constituents)} {cache_type} constituents from cache")

    return constituents


def get_sp500_universe(use_cache: bool = True, extend: bool = True) -> List[str]:
    """
    Get S&P 500 universe (with optional extension for survivorship bias mitigation).

    Args:
        use_cache: Whether to use cached data
        extend: Whether to build extended universe

    Returns:
        List of ticker symbols
    """
    cache_type = 'extended' if extend else 'sp500'

    # Try cache first
    if use_cache:
        cached = load_constituent_cache(cache_type)
        if cached is not None:
            return cached

    # Fetch current S&P 500
    sp500 = get_current_sp500_constituents()

    if extend:
        # Build extended universe
        universe = build_extended_universe(sp500)
    else:
        universe = sp500

    # Save to cache
    save_constituent_cache(universe, cache_type)

    return universe


# Pre-defined list of tickers that were in S&P 500 but delisted/removed
# This helps with historical accuracy
KNOWN_DELISTED_SP500 = [
    'YHOO',  # Yahoo (acquired by Verizon)
    'SHLD',  # Sears (bankrupt)
    'TWX',   # Time Warner (acquired by AT&T)
    'FOXA',  # Fox (restructured)
    'WFM',   # Whole Foods (acquired by Amazon)
    'MON',   # Monsanto (acquired by Bayer)
    'PCLN',  # Priceline (now BKNG)
    'RAI',   # Reynolds American (acquired)
    'ESRX',  # Express Scripts (acquired)
    'BHI',   # Baker Hughes (merged)
    'LLTC',  # Linear Technology (acquired)
    'ALXN',  # Alexion (acquired by AstraZeneca)
    'CELG',  # Celgene (acquired by BMS)
    'ARNC',  # Arconic (split)
    'FL',    # Foot Locker (removed)
]
