"""
Fundamental data extraction for value signals (B/M, E/P ratios).

Note: yfinance provides current fundamentals, not historical point-in-time data.
We use quarterly snapshots with caching and forward-fill as a practical workaround.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional
import time
import os
import logging
from datetime import datetime

from config.parameters import cache_config, data_config
from utils.cache import save_to_pickle, load_from_pickle, save_to_json, load_from_json

logger = logging.getLogger(__name__)


class FundamentalDataManager:
    """Manages fundamental data extraction from yfinance."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize FundamentalDataManager.

        Args:
            cache_dir: Directory for caching fundamental data
        """
        self.cache_dir = cache_dir or cache_config.FUNDAMENTAL_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, ticker: str, date: str) -> str:
        """Get cache file path for ticker and date."""
        return os.path.join(self.cache_dir, f"{ticker}_{date}.pkl")

    def _fetch_ticker_info(self, ticker: str) -> Optional[Dict]:
        """
        Fetch fundamental info for single ticker with retry logic.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with fundamental data or None
        """
        for attempt in range(data_config.MAX_RETRIES):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if info and len(info) > 0:
                    return info

            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                time.sleep(data_config.API_DELAY_SECONDS)

        logger.warning(f"Could not fetch fundamental data for {ticker}")
        return None

    def get_book_to_market(self, tickers: List[str],
                          date: Optional[str] = None) -> pd.Series:
        """
        Calculate Book-to-Market ratio.

        B/M = Book Value per Share / Price per Share

        Args:
            tickers: List of ticker symbols
            date: Date for calculation (currently uses latest available)

        Returns:
            Series with ticker as index, B/M as values
        """
        logger.info(f"Fetching Book-to-Market ratios for {len(tickers)} tickers")

        bm_ratios = {}

        for ticker in tickers:
            try:
                info = self._fetch_ticker_info(ticker)

                if info and 'bookValue' in info and 'currentPrice' in info:
                    book_value = info['bookValue']
                    price = info['currentPrice']

                    if price and price > 0 and book_value:
                        bm_ratio = book_value / price
                        bm_ratios[ticker] = bm_ratio

                time.sleep(0.1)  # Gentle rate limiting

            except Exception as e:
                logger.debug(f"Error getting B/M for {ticker}: {e}")

        logger.info(f"Successfully retrieved B/M ratios for {len(bm_ratios)} tickers")

        return pd.Series(bm_ratios)

    def get_earnings_to_price(self, tickers: List[str],
                             date: Optional[str] = None) -> pd.Series:
        """
        Calculate Earnings-to-Price ratio (inverse of P/E).

        E/P = 1 / P/E

        Args:
            tickers: List of ticker symbols
            date: Date for calculation (currently uses latest available)

        Returns:
            Series with ticker as index, E/P as values
        """
        logger.info(f"Fetching Earnings-to-Price ratios for {len(tickers)} tickers")

        ep_ratios = {}

        for ticker in tickers:
            try:
                info = self._fetch_ticker_info(ticker)

                if info and 'trailingPE' in info:
                    pe_ratio = info['trailingPE']

                    if pe_ratio and pe_ratio > 0:
                        ep_ratio = 1.0 / pe_ratio
                        ep_ratios[ticker] = ep_ratio

                time.sleep(0.1)  # Gentle rate limiting

            except Exception as e:
                logger.debug(f"Error getting E/P for {ticker}: {e}")

        logger.info(f"Successfully retrieved E/P ratios for {len(ep_ratios)} tickers")

        return pd.Series(ep_ratios)

    def get_market_cap(self, tickers: List[str]) -> pd.Series:
        """
        Get market capitalization for tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Series with market caps
        """
        logger.info(f"Fetching market caps for {len(tickers)} tickers")

        market_caps = {}

        for ticker in tickers:
            try:
                info = self._fetch_ticker_info(ticker)

                if info and 'marketCap' in info:
                    market_cap = info['marketCap']
                    if market_cap:
                        market_caps[ticker] = market_cap

                time.sleep(0.1)

            except Exception as e:
                logger.debug(f"Error getting market cap for {ticker}: {e}")

        return pd.Series(market_caps)

    def get_fundamentals_batch(self, tickers: List[str],
                              date: Optional[str] = None) -> pd.DataFrame:
        """
        Batch download fundamentals for multiple tickers.

        Returns DataFrame with columns: [B/M, E/P, market_cap]

        Note: Due to yfinance limitations, this returns current data.
        For backtesting, we cache quarterly snapshots and forward-fill.

        Args:
            tickers: List of tickers
            date: Date for snapshot (used for caching)

        Returns:
            DataFrame with fundamental ratios
        """
        logger.info(f"Fetching fundamental data for {len(tickers)} tickers")

        # Get B/M ratios
        bm = self.get_book_to_market(tickers, date)

        # Get E/P ratios
        ep = self.get_earnings_to_price(tickers, date)

        # Get market caps
        mc = self.get_market_cap(tickers)

        # Combine into DataFrame
        fundamentals = pd.DataFrame({
            'B/M': bm,
            'E/P': ep,
            'market_cap': mc
        })

        logger.info(f"Retrieved fundamentals for {len(fundamentals)} tickers")

        return fundamentals

    def cache_quarterly_snapshot(self, tickers: List[str],
                                date: str) -> pd.DataFrame:
        """
        Cache quarterly fundamental snapshot.

        Args:
            tickers: List of tickers
            date: Snapshot date (YYYY-MM-DD)

        Returns:
            DataFrame with fundamentals
        """
        logger.info(f"Creating quarterly fundamental snapshot for {date}")

        # Get fundamentals
        fundamentals = self.get_fundamentals_batch(tickers, date)

        # Save to cache
        cache_file = os.path.join(self.cache_dir, f"snapshot_{date}.pkl")
        save_to_pickle(fundamentals, cache_file)

        logger.info(f"Saved snapshot to {cache_file}")

        return fundamentals

    def load_or_create_snapshot(self, tickers: List[str],
                               date: str,
                               use_cache: bool = True) -> pd.DataFrame:
        """
        Load snapshot from cache or create if doesn't exist.

        Args:
            tickers: List of tickers
            date: Snapshot date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with fundamentals
        """
        cache_file = os.path.join(self.cache_dir, f"snapshot_{date}.pkl")

        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading snapshot from cache: {date}")
            return load_from_pickle(cache_file)

        # Create new snapshot
        return self.cache_quarterly_snapshot(tickers, date)

    def get_fundamentals_timeline(self, tickers: List[str],
                                  start_date: str,
                                  end_date: str) -> pd.DataFrame:
        """
        Build fundamental data timeline using quarterly snapshots.

        Strategy:
        1. Create snapshots at quarterly intervals
        2. Forward-fill between snapshots
        3. Return daily timeline

        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with MultiIndex (date, ticker) and fundamental data
        """
        logger.info(f"Building fundamental timeline from {start_date} to {end_date}")

        # Generate quarterly dates
        quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        quarterly_dates = [d.strftime('%Y-%m-%d') for d in quarterly_dates]

        # Collect quarterly snapshots
        snapshots = {}
        for qdate in quarterly_dates:
            snapshot = self.load_or_create_snapshot(tickers, qdate)
            snapshots[qdate] = snapshot

        # Combine snapshots into timeline
        if not snapshots:
            return pd.DataFrame()

        # Create multi-index DataFrame
        timeline_data = []
        for date, snapshot in snapshots.items():
            for ticker in snapshot.index:
                row = snapshot.loc[ticker].to_dict()
                row['date'] = date
                row['ticker'] = ticker
                timeline_data.append(row)

        timeline_df = pd.DataFrame(timeline_data)

        if len(timeline_df) > 0:
            timeline_df['date'] = pd.to_datetime(timeline_df['date'])
            timeline_df = timeline_df.set_index(['date', 'ticker'])

        logger.info(f"Built timeline with {len(quarterly_dates)} snapshots")

        return timeline_df

    def forward_fill_fundamentals(self, timeline_df: pd.DataFrame,
                                  daily_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Forward-fill quarterly fundamental data to daily frequency.

        Args:
            timeline_df: DataFrame with quarterly snapshots
            daily_dates: Daily date range to fill

        Returns:
            DataFrame with daily fundamental data
        """
        if len(timeline_df) == 0:
            return pd.DataFrame()

        # Unstack to get tickers as columns
        unstacked = timeline_df.unstack(level='ticker')

        # Reindex to daily dates and forward-fill
        daily = unstacked.reindex(daily_dates, method='ffill')

        # Re-stack to original format
        stacked = daily.stack(level=-1, future_stack=True)

        return stacked
