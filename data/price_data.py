"""
Price data download and caching with batch yfinance downloads.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Optional, Dict
import time
import os
import logging

from config.parameters import data_config, cache_config, universe_config
from utils.cache import save_to_pickle, load_from_pickle, is_cache_valid, save_to_json, load_from_json
from utils.helpers import forward_fill_limited, remove_outliers

logger = logging.getLogger(__name__)


class PriceDataManager:
    """Manages price data download and caching with intelligent batch processing."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize PriceDataManager.

        Args:
            cache_dir: Directory for caching price data
        """
        self.cache_dir = cache_dir or cache_config.PRICE_CACHE_DIR
        self.metadata_file = os.path.join(self.cache_dir, 'metadata.json')

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        metadata = load_from_json(self.metadata_file)
        if metadata is None:
            metadata = {}
        return metadata

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        save_to_json(self.metadata, self.metadata_file)

    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for ticker."""
        return os.path.join(self.cache_dir, f"{ticker}.pkl")

    def _cache_exists(self, ticker: str) -> bool:
        """Check if cached data exists for ticker."""
        cache_path = self._get_cache_path(ticker)
        return os.path.exists(cache_path)

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load cached price data for ticker."""
        cache_path = self._get_cache_path(ticker)
        return load_from_pickle(cache_path)

    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Save price data to cache and update metadata."""
        cache_path = self._get_cache_path(ticker)
        save_to_pickle(data, cache_path)

        # Update metadata
        if len(data) > 0:
            self.metadata[ticker] = {
                'start_date': str(data.index.min().date()),
                'end_date': str(data.index.max().date()),
                'num_records': len(data),
                'last_updated': str(pd.Timestamp.now())
            }
            self._save_metadata()

    def _get_cache_date_range(self, ticker: str) -> Optional[Tuple[str, str]]:
        """Get date range of cached data."""
        if ticker in self.metadata:
            return (self.metadata[ticker]['start_date'],
                   self.metadata[ticker]['end_date'])
        return None

    def download_prices(self,
                       tickers: List[str],
                       start: str,
                       end: str,
                       force_refresh: bool = False) -> pd.DataFrame:
        """
        Download price data with intelligent caching.

        Strategy:
        - Check cache for existing data
        - Download only missing tickers or date ranges
        - Use batch downloads for efficiency
        - Handle errors gracefully

        Args:
            tickers: List of ticker symbols
            start: Start date ('YYYY-MM-DD')
            end: End date ('YYYY-MM-DD')
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with MultiIndex (date, ticker) or DateIndex with ticker columns
        """
        logger.info(f"Downloading price data for {len(tickers)} tickers from {start} to {end}")

        all_data = {}
        tickers_to_download = []

        # Check cache for each ticker
        for ticker in tickers:
            if force_refresh or not self._cache_exists(ticker):
                tickers_to_download.append(ticker)
            else:
                # Load from cache
                cached_data = self._load_from_cache(ticker)
                if cached_data is not None and len(cached_data) > 0:
                    # Check if cache covers required date range
                    cache_range = self._get_cache_date_range(ticker)
                    if cache_range:
                        cache_start, cache_end = cache_range
                        if cache_start <= start and cache_end >= end:
                            # Cache is sufficient
                            all_data[ticker] = cached_data.loc[start:end]
                            continue

                # Need to download
                tickers_to_download.append(ticker)

        logger.info(f"Found {len(all_data)} tickers in cache, need to download {len(tickers_to_download)}")

        # Download missing tickers in batches
        if tickers_to_download:
            downloaded = self._batch_download(tickers_to_download, start, end)
            all_data.update(downloaded)

        # Combine all data
        if not all_data:
            logger.warning("No data downloaded successfully")
            return pd.DataFrame()

        # Create combined DataFrame with tickers as columns
        combined = pd.DataFrame(all_data)

        logger.info(f"Successfully retrieved data for {len(combined.columns)} tickers")

        return combined

    def _batch_download(self,
                       tickers: List[str],
                       start: str,
                       end: str) -> Dict[str, pd.DataFrame]:
        """
        Download tickers in batches to handle yfinance limits.

        Args:
            tickers: List of tickers to download
            start: Start date
            end: End date

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        all_data = {}
        batch_size = data_config.BATCH_SIZE

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tickers) + batch_size - 1) // batch_size

            logger.info(f"Downloading batch {batch_num}/{total_batches} ({len(batch)} tickers)")

            try:
                # Download batch
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    progress=False,
                    group_by='ticker',
                    threads=True,
                    auto_adjust=False  # We'll use Adj Close manually
                )

                # Process each ticker in batch
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            # Single ticker - data is DataFrame
                            ticker_data = data
                        else:
                            # Multiple tickers - data is multi-level columns
                            if ticker in data.columns.get_level_values(0):
                                ticker_data = data[ticker]
                            else:
                                logger.warning(f"No data returned for {ticker}")
                                continue

                        # Extract Adjusted Close prices
                        if 'Adj Close' in ticker_data.columns:
                            prices = ticker_data['Adj Close'].dropna()

                            if len(prices) > 0:
                                # Clean data
                                prices = self._clean_price_data(prices, ticker)

                                # Save to cache
                                self._save_to_cache(ticker, prices.to_frame('Adj Close'))

                                # Store in results
                                all_data[ticker] = prices

                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")

                # Rate limiting
                if i + batch_size < len(tickers):
                    time.sleep(data_config.API_DELAY_SECONDS)

            except Exception as e:
                logger.error(f"Batch download failed: {e}")
                # Try individual downloads as fallback
                all_data.update(self._fallback_individual_download(batch, start, end))

        return all_data

    def _fallback_individual_download(self,
                                     tickers: List[str],
                                     start: str,
                                     end: str) -> Dict[str, pd.DataFrame]:
        """
        Fallback to individual ticker downloads if batch fails.

        Args:
            tickers: List of tickers
            start: Start date
            end: End date

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        logger.info(f"Falling back to individual downloads for {len(tickers)} tickers")

        all_data = {}

        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)

                if len(data) > 0 and 'Adj Close' in data.columns:
                    prices = data['Adj Close'].dropna()
                    prices = self._clean_price_data(prices, ticker)

                    self._save_to_cache(ticker, prices.to_frame('Adj Close'))
                    all_data[ticker] = prices

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")

        return all_data

    def _clean_price_data(self, prices: pd.Series, ticker: str) -> pd.Series:
        """
        Clean price data by removing outliers and handling issues.

        Args:
            prices: Price series
            ticker: Ticker symbol

        Returns:
            Cleaned price series
        """
        # Remove prices below minimum threshold (penny stocks / data errors)
        if universe_config.MIN_PRICE > 0:
            low_prices = prices < universe_config.MIN_PRICE
            if low_prices.sum() > len(prices) * 0.5:
                # If more than half the data is below threshold, likely a valid penny stock
                pass
            else:
                # Filter out occasional low prices (likely errors)
                prices = prices[prices >= universe_config.MIN_PRICE]

        # Calculate returns to identify outliers
        returns = prices.pct_change()

        # Flag extreme returns (> 100% in single day = likely data error)
        extreme_returns = returns.abs() > 1.0
        if extreme_returns.sum() > 0:
            logger.warning(f"{ticker}: Found {extreme_returns.sum()} extreme returns, removing")
            # Remove days with extreme returns
            prices = prices[~extreme_returns]

        # Forward fill missing values (limited)
        if prices.isna().sum() > 0:
            prices = forward_fill_limited(prices.to_frame(), max_periods=data_config.MAX_FFILL_DAYS).iloc[:, 0]

        return prices

    def get_adjusted_prices(self,
                           tickers: List[str],
                           start: str,
                           end: str,
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Get adjusted close prices for analysis.

        Args:
            tickers: List of tickers
            start: Start date
            end: End date
            force_refresh: Force refresh cache

        Returns:
            DataFrame with dates as index, tickers as columns
        """
        data = self.download_prices(tickers, start, end, force_refresh)

        # If data is already in correct format (dates × tickers), return it
        if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
            return data

        # Otherwise, it might be multi-index, pivot it
        if isinstance(data.index, pd.MultiIndex):
            data = data.unstack(level=1)

        return data

    def calculate_returns(self,
                         prices: pd.DataFrame,
                         periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns from prices.

        Args:
            prices: DataFrame with prices
            periods: Number of periods for return calculation

        Returns:
            DataFrame with returns
        """
        returns = prices.pct_change(periods=periods)
        return returns

    def get_returns(self,
                   tickers: List[str],
                   start: str,
                   end: str,
                   periods: int = 1) -> pd.DataFrame:
        """
        Get returns for tickers.

        Args:
            tickers: List of tickers
            start: Start date
            end: End date
            periods: Number of periods for return calculation

        Returns:
            DataFrame with returns
        """
        prices = self.get_adjusted_prices(tickers, start, end)
        returns = self.calculate_returns(prices, periods)
        return returns

    def get_volume_data(self,
                       tickers: List[str],
                       start: str,
                       end: str) -> pd.DataFrame:
        """
        Get trading volume data.

        Args:
            tickers: List of tickers
            start: Start date
            end: End date

        Returns:
            DataFrame with volume data
        """
        logger.info(f"Downloading volume data for {len(tickers)} tickers")

        all_volume = {}

        for ticker in tickers:
            try:
                # Check cache first
                cache_path = self._get_cache_path(ticker)
                cached = load_from_pickle(cache_path)

                if cached is not None and 'Volume' in cached.columns:
                    all_volume[ticker] = cached['Volume']
                else:
                    # Download fresh data
                    data = yf.download(ticker, start=start, end=end, progress=False)
                    if len(data) > 0 and 'Volume' in data.columns:
                        all_volume[ticker] = data['Volume']

            except Exception as e:
                logger.error(f"Error getting volume for {ticker}: {e}")

        return pd.DataFrame(all_volume)

    def get_market_caps(self,
                       tickers: List[str],
                       date: Optional[str] = None) -> pd.Series:
        """
        Get market capitalizations for tickers.

        Note: yfinance provides current market cap, not historical.
        This is a limitation we acknowledge.

        Args:
            tickers: List of tickers
            date: Date for market cap (currently ignored, returns latest)

        Returns:
            Series with market caps
        """
        logger.info(f"Fetching market caps for {len(tickers)} tickers")

        market_caps = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if 'marketCap' in info and info['marketCap']:
                    market_caps[ticker] = info['marketCap']

            except Exception as e:
                logger.debug(f"Could not get market cap for {ticker}: {e}")

        return pd.Series(market_caps)

    def validate_data_quality(self, prices: pd.DataFrame) -> Dict:
        """
        Validate quality of price data.

        Args:
            prices: Price DataFrame

        Returns:
            Dictionary with validation results
        """
        validation = {
            'total_tickers': len(prices.columns),
            'date_range': (prices.index.min(), prices.index.max()),
            'total_days': len(prices),
            'missing_data_pct': prices.isna().sum().sum() / (len(prices) * len(prices.columns)) * 100,
            'tickers_with_insufficient_data': [],
        }

        # Check each ticker
        for ticker in prices.columns:
            ticker_data = prices[ticker].dropna()
            if len(ticker_data) < universe_config.MIN_HISTORY_DAYS:
                validation['tickers_with_insufficient_data'].append(ticker)

        validation['tickers_with_sufficient_data'] = (
            validation['total_tickers'] - len(validation['tickers_with_insufficient_data'])
        )

        return validation
