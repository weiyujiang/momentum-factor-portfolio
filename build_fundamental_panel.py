"""
Build a historical fundamental panel (B/M, E/P, market cap) for the universe.

Methodology:
- Source: yfinance quarterly_balance_sheet + quarterly_financials
- Reporting lag: 45 days (SEC 10-Q deadline) to approximate point-in-time data
- Forward-fill across trading days between quarter-end filings
- Saves panel to cache/fundamentals/fundamental_panel.pkl
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from tqdm import tqdm

from data.sp500_constituents import get_sp500_universe
from data.price_data import PriceDataManager

PANEL_PATH  = 'cache/fundamentals/fundamental_panel.pkl'
LAG_DAYS    = 45   # reporting delay: quarterly data available ~45 days after quarter end
BATCH_PAUSE = 0.3  # seconds between tickers

os.makedirs('cache/fundamentals', exist_ok=True)

# ── Universe & prices ─────────────────────────────────────────────────────────
print("Loading universe and prices...")
universe = get_sp500_universe(use_cache=True, extend=True)
pm       = PriceDataManager()
prices   = pm.get_adjusted_prices(universe, '2019-01-01', '2026-04-25')
tickers  = list(prices.columns)
print(f"Universe: {len(tickers)} tickers | Price dates: {len(prices)}")

# ── Download quarterly fundamentals ──────────────────────────────────────────
print("\nDownloading quarterly fundamentals (this takes ~3-5 min)...")

records = []   # list of dicts: {ticker, report_date, avail_date, book_value_ps, net_income_ttm, shares}

failed = []
for ticker in tqdm(tickers, desc="Tickers"):
    try:
        t = yf.Ticker(ticker)

        # ── Balance sheet (book value = Common Stock Equity) ─────────────
        bs = t.quarterly_balance_sheet
        if bs is None or bs.empty:
            failed.append(ticker)
            continue

        # Find equity row
        eq_row = None
        for candidate in ['Common Stock Equity', 'Stockholders Equity',
                           'Total Stockholder Equity', 'Total Equity']:
            if candidate in bs.index:
                eq_row = bs.loc[candidate]
                break
        if eq_row is None:
            failed.append(ticker)
            time.sleep(BATCH_PAUSE)
            continue

        # Shares outstanding — try balance sheet rows first
        sh_row = None
        for candidate in ['Ordinary Shares Number', 'Share Issued',
                           'Common Stock', 'Shares Outstanding']:
            if candidate in bs.index:
                sh_row = bs.loc[candidate]
                break

        # Pre-fetch a fallback shares count once per ticker (avoid per-quarter API calls)
        shares_fallback = np.nan
        try:
            fi_sh = getattr(t.fast_info, 'shares', None)
            if fi_sh and float(fi_sh) > 0:
                shares_fallback = float(fi_sh)
        except Exception:
            pass
        if np.isnan(shares_fallback):
            try:
                info = t.info
                s = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
                if s and float(s) > 0:
                    shares_fallback = float(s)
            except Exception:
                pass

        # ── Income statement (net income for E/P) ────────────────────────
        fi = t.quarterly_financials
        ni_row = None
        if fi is not None and not fi.empty:
            for candidate in ['Net Income',
                               'Net Income From Continuing And Discontinued Operation',
                               'Net Income From Continuing Operation Net Minority Interest',
                               'Net Income Common Stockholders']:
                if candidate in fi.index:
                    ni_row = fi.loc[candidate]
                    break

        # ── Build per-quarter records ─────────────────────────────────────
        for qdate in eq_row.index:
            try:
                qdate = pd.Timestamp(qdate)
                eq_val = float(eq_row.get(qdate, np.nan))
                if np.isnan(eq_val):
                    continue

                # Shares: prefer balance sheet row, then pre-fetched fallback
                sh_val = np.nan
                if sh_row is not None:
                    try:
                        sh_val = float(sh_row.get(qdate, np.nan))
                    except Exception:
                        pass
                if np.isnan(sh_val) or sh_val <= 0:
                    sh_val = shares_fallback

                book_value_ps = eq_val / sh_val if (sh_val > 0 and not np.isnan(sh_val)) else np.nan

                # Net income TTM (sum of 4 most recent quarters ending at qdate)
                ni_ttm = np.nan
                if ni_row is not None:
                    ni_cols = [c for c in ni_row.index if pd.Timestamp(c) <= qdate]
                    ni_cols_sorted = sorted(ni_cols, reverse=True)[:4]
                    if len(ni_cols_sorted) >= 1:
                        ni_ttm = float(ni_row.loc[ni_cols_sorted].sum())

                # Data becomes available LAG_DAYS after quarter end
                avail_date = qdate + pd.Timedelta(days=LAG_DAYS)

                records.append({
                    'ticker':       ticker,
                    'report_date':  qdate,
                    'avail_date':   avail_date,
                    'equity':       eq_val,
                    'shares':       sh_val if not np.isnan(sh_val) else np.nan,
                    'book_value_ps': book_value_ps,
                    'ni_ttm':       ni_ttm,
                })
            except Exception:
                continue

        time.sleep(BATCH_PAUSE)

    except Exception as e:
        failed.append(ticker)
        time.sleep(BATCH_PAUSE)

print(f"\nDownloaded: {len(tickers)-len(failed)} tickers OK | {len(failed)} failed")
print(f"Total records: {len(records)}")
if failed:
    print(f"Failed tickers: {failed[:10]}{'...' if len(failed)>10 else ''}")

if not records:
    print("No fundamental data downloaded. Exiting.")
    sys.exit(1)

raw = pd.DataFrame(records)
raw['avail_date'] = pd.to_datetime(raw['avail_date'])
raw['report_date'] = pd.to_datetime(raw['report_date'])

# ── Build daily B/M and E/P panel ────────────────────────────────────────────
print("\nBuilding daily fundamental panel with 45-day lag...")

trading_days = prices.index  # already a DatetimeIndex

# For each trading day, for each ticker: find most recent available quarter
bm_panel = pd.DataFrame(np.nan, index=trading_days, columns=tickers)
ep_panel = pd.DataFrame(np.nan, index=trading_days, columns=tickers)
mc_panel = pd.DataFrame(np.nan, index=trading_days, columns=tickers)

for ticker in tqdm(tickers, desc="Building panel"):
    tk_data = raw[raw['ticker'] == ticker].sort_values('avail_date')
    if tk_data.empty:
        continue

    tk_prices = prices[ticker].dropna()

    for day in trading_days:
        # Most recent quarter whose data was available on or before `day`
        available = tk_data[tk_data['avail_date'] <= day]
        if available.empty:
            continue
        latest = available.iloc[-1]

        # Price on this day
        price = tk_prices.get(day, np.nan)
        if np.isnan(price) or price <= 0:
            continue

        # B/M = book value per share / price
        bv_ps = latest['book_value_ps']
        if not np.isnan(bv_ps):
            bm_panel.loc[day, ticker] = bv_ps / price

        # E/P = net income TTM / market cap  (ni_ttm / (price * shares))
        ni_ttm = latest['ni_ttm']
        shares = latest['shares']
        if not np.isnan(ni_ttm) and not np.isnan(shares) and shares > 0:
            mktcap = price * shares
            if mktcap > 0:
                ep_panel.loc[day, ticker] = ni_ttm / mktcap
                mc_panel.loc[day, ticker] = mktcap

panel = {
    'bm': bm_panel,
    'ep': ep_panel,
    'mc': mc_panel,
    'raw': raw,
}

with open(PANEL_PATH, 'wb') as f:
    pickle.dump(panel, f)

print(f"\nPanel saved: {PANEL_PATH}")
print(f"B/M coverage (% non-NaN per day, mean): {bm_panel.notna().mean(axis=1).mean()*100:.1f}%")
print(f"E/P coverage (% non-NaN per day, mean): {ep_panel.notna().mean(axis=1).mean()*100:.1f}%")
print(f"Market cap coverage:                     {mc_panel.notna().mean(axis=1).mean()*100:.1f}%")

# Preview
print("\nSample B/M (last 3 dates, first 5 tickers):")
print(bm_panel.iloc[-3:, :5].round(3).to_string())
print("\nSample E/P (last 3 dates, first 5 tickers):")
print(ep_panel.iloc[-3:, :5].round(4).to_string())
