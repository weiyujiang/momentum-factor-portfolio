"""
Backtest using real fundamental data (B/M and E/P) for the value signal.
Compares: Value-Hedged (real fundamentals) vs Pure Momentum vs S&P 500.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['BACKTEST_START'] = '2020-01-01'
os.environ['BACKTEST_END']   = '2026-04-25'
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.parameters import backtest_config, portfolio_config, risk_config
from data.sp500_constituents import get_sp500_universe
from data.price_data import PriceDataManager
from data.market_data import get_market_index
from signals.momentum import calculate_momentum_1221, calculate_momentum_dispersion
from portfolio.construction import PortfolioConstructor
from portfolio.rebalancing import Rebalancer
from backtest.performance import PerformanceTracker
from utils.helpers import get_month_end_dates
from utils.logging_config import setup_logging

setup_logging(log_level='WARNING', log_file='results/fundamental_backtest.log')
os.makedirs('results/fundamental', exist_ok=True)

PANEL_PATH = 'cache/fundamentals/fundamental_panel.pkl'
RF = 0.02

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading price data and fundamental panel...")
universe = get_sp500_universe(use_cache=True, extend=True)
pm       = PriceDataManager()
prices   = pm.get_adjusted_prices(universe, '2020-01-01', '2026-04-25')
mom_all  = calculate_momentum_1221(prices)

with open('cache/vix_data.pkl', 'rb') as f:
    vix_data = pickle.load(f)

with open(PANEL_PATH, 'rb') as f:
    panel = pickle.load(f)

bm_panel = panel['bm']
ep_panel = panel['ep']
mc_panel = panel['mc']

print(f"Prices: {prices.shape} | B/M coverage: {bm_panel.notna().mean(axis=1).mean()*100:.1f}%")

# ── Value signal ──────────────────────────────────────────────────────────────
def compute_value_score(date_str):
    """
    Composite value score = Z(B/M) + Z(E/P), cross-sectional.
    Higher score = cheaper stock.
    Returns Series indexed by ticker.
    """
    if date_str not in bm_panel.index:
        return pd.Series(dtype=float)

    bm = bm_panel.loc[date_str].dropna()
    ep = ep_panel.loc[date_str].dropna()

    common = bm.index.intersection(ep.index)
    if len(common) < 10:
        # Fall back to whichever signal has more data
        common = bm.index if len(bm) >= len(ep) else ep.index

    bm_z = pd.Series(dtype=float)
    ep_z = pd.Series(dtype=float)

    if len(bm) > 2:
        bm_z = (bm - bm.mean()) / bm.std()
    if len(ep) > 2:
        ep_z = (ep - ep.mean()) / ep.std()

    # Winsorise at ±3σ
    bm_z = bm_z.clip(-3, 3)
    ep_z = ep_z.clip(-3, 3)

    combined = bm_z.add(ep_z, fill_value=0)
    return combined


def compute_value_spread(date_str):
    """
    Value spread = 90th pct value score - 10th pct value score.
    High spread -> bubble conditions -> elevated crash risk.
    """
    vs = compute_value_score(date_str)
    if len(vs) < 20:
        return np.nan
    return float(vs.quantile(0.9) - vs.quantile(0.1))


# ── Calibrate crash risk threshold ───────────────────────────────────────────
print("Calibrating crash risk thresholds from 2021-2022 training data...")
train_dates = [d for d in mom_all.index
               if '2021-03-01' <= d.strftime('%Y-%m-%d') <= '2022-12-31']

disps   = [calculate_momentum_dispersion(mom_all, d.strftime('%Y-%m-%d'))
           for d in train_dates]
disps   = [x for x in disps if not np.isnan(x)]
spreads = [compute_value_spread(d.strftime('%Y-%m-%d')) for d in train_dates]
spreads = [x for x in spreads if not np.isnan(x)]

DISP_THR   = float(np.percentile(disps,   75)) if disps   else 1.32
SPREAD_THR = float(np.percentile(spreads, 75)) if spreads else 2.0

print(f"  Dispersion threshold (75th pct): {DISP_THR:.4f}")
print(f"  Value spread threshold (75th pct): {SPREAD_THR:.4f}")


# ── Core backtest engine ──────────────────────────────────────────────────────
def run_backtest(start_date, end_date,
                 alpha=0.70, n_stocks=50,
                 use_hedge=True, use_value=True,
                 disp_thr=None, spread_thr=None,
                 vix_thr=25.0):

    disp_thr   = disp_thr   or DISP_THR
    spread_thr = spread_thr or SPREAD_THR

    pc   = PortfolioConstructor()
    reb  = Rebalancer()
    perf = PerformanceTracker()

    port_value   = backtest_config.INITIAL_CAPITAL
    curr_weights = pd.Series(dtype=float)

    rebal_dates = get_month_end_dates(start_date, end_date)

    for i, rebal_date in enumerate(rebal_dates):
        date_str = rebal_date.strftime('%Y-%m-%d')

        if date_str not in prices.index or date_str not in mom_all.index:
            continue

        mom_scores = mom_all.loc[date_str].dropna()
        if len(mom_scores) == 0:
            continue

        # ── Stock selection ───────────────────────────────────────────────
        if use_value and alpha < 1.0:
            val_scores = compute_value_score(date_str)
            if len(val_scores) >= 10:
                # Compute percentile ranks for both signals
                mom_pct = mom_scores.rank(pct=True)
                val_pct = val_scores.rank(pct=True)
                common  = mom_pct.index.intersection(val_pct.index)

                if len(common) >= 10:
                    combined = alpha * mom_pct.loc[common] + (1 - alpha) * val_pct.loc[common]
                    # Value filter: exclude bottom 50% of value universe (expensive stocks)
                    value_eligible = val_pct.loc[common][val_pct.loc[common] >= 0.50].index
                    combined = combined.loc[combined.index.intersection(value_eligible)]
                    selected = combined.nlargest(n_stocks).index.tolist()
                else:
                    selected = mom_scores.nlargest(n_stocks).index.tolist()
            else:
                selected = mom_scores.nlargest(n_stocks).index.tolist()
        else:
            # Pure momentum
            selected = mom_scores.nlargest(n_stocks).index.tolist()

        if not selected:
            continue

        # ── Weighting (market cap if available, else equal) ───────────────
        mc_today = mc_panel.loc[date_str] if date_str in mc_panel.index else pd.Series(dtype=float)
        mc_sel   = mc_today.reindex(selected).dropna()

        if len(mc_sel) >= len(selected) * 0.5:
            # Market-cap weight with 5% cap
            w = mc_sel / mc_sel.sum()
            w = w.clip(upper=portfolio_config.MAX_POSITION_SIZE)
            w = w / w.sum()
            # Pad missing tickers with equal weight residual
            missing = [t for t in selected if t not in w.index]
            if missing:
                residual = (1 - w.sum()) / len(missing)
                for t in missing:
                    w[t] = residual
            target_weights = w
        else:
            target_weights = pc.construct_long_portfolio(
                selected_tickers=selected, fundamentals=None,
                prices=prices, signals=mom_scores)

        # ── Crash-risk hedge ──────────────────────────────────────────────
        hedge_alloc = 0.0
        if use_hedge:
            vix_val  = float(vix_data.loc[date_str, 'VIX']) if date_str in vix_data.index else 25.0
            vix_sig  = max(0.0, (vix_val - vix_thr) / vix_thr)

            disp_raw = calculate_momentum_dispersion(mom_all, date_str)
            disp_sig = 0.0 if np.isnan(disp_raw) else max(0.0, (disp_raw - disp_thr) / disp_thr)

            spread_raw = compute_value_spread(date_str)
            spread_sig = 0.0 if np.isnan(spread_raw) else max(0.0, (spread_raw - spread_thr) / spread_thr)

            composite   = 0.40 * vix_sig + 0.30 * disp_sig + 0.30 * spread_sig
            hedge_alloc = min(composite * risk_config.HEDGE_ALLOCATION_MAX,
                              risk_config.HEDGE_ALLOCATION_MAX)

            target_weights = pc.apply_hedge(target_weights, hedge_alloc)

        new_weights, costs = reb.execute_rebalance(curr_weights, target_weights, port_value)
        curr_weights = new_weights
        port_value  -= costs
        perf.record_rebalance(date_str, 0, costs, hedge_alloc)

        # Mark to market
        next_date = (rebal_dates[i+1] if i < len(rebal_dates)-1
                     else pd.Timestamp(end_date))
        for dt in prices.loc[rebal_date:next_date].index:
            if len(curr_weights) == 0:
                continue
            r    = ((prices.loc[dt] - prices.shift(1).loc[dt]) /
                    prices.shift(1).loc[dt]).reindex(curr_weights.index).fillna(0)
            pr   = float((curr_weights * r).sum())
            port_value *= (1 + pr)
            ds = dt.strftime('%Y-%m-%d')
            perf.record_daily_value(ds, port_value)
            perf.record_daily_return(ds, pr)

    return pd.DataFrame({'portfolio_value': perf.get_value_series(),
                         'returns':         perf.get_returns_series()})


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(rets, vals, bench=None):
    rets = rets.dropna(); vals = vals.dropna()
    n    = len(rets)
    if n < 20: return {}
    total   = float(vals.iloc[-1]/vals.iloc[0] - 1)
    ann_r   = float((1+total)**(252/n) - 1)
    ann_v   = float(rets.std() * np.sqrt(252))
    sharpe  = float((ann_r - RF) / ann_v) if ann_v > 0 else np.nan
    ds      = rets[rets < 0]
    sortino = float((ann_r - RF) / (ds.std()*np.sqrt(252))) if len(ds)>0 else np.nan
    pk      = vals.expanding().max()
    mdd     = float(((vals-pk)/pk).min())
    calmar  = float(ann_r/abs(mdd)) if mdd else np.nan
    cvar    = float(-rets[rets<=rets.quantile(0.05)].mean())
    win     = float((rets>0).mean())
    ir      = np.nan
    if bench is not None:
        bench = bench[~bench.index.duplicated()]
        common = rets.index.intersection(bench.index)
        if len(common) > 30:
            ex = rets.loc[common] - bench.loc[common]
            ae = float((1+ex).prod()**(252/len(ex))-1)
            te = float(ex.std()*np.sqrt(252))
            ir = float(ae/te) if te > 0 else np.nan
    bear_2022 = rets.loc['2022-01-03':'2022-10-13'] if '2022-01-03' in rets.index else pd.Series()
    bear_ret  = float((1+bear_2022).prod()-1) if len(bear_2022) > 0 else np.nan
    return dict(total=total, ann_return=ann_r, volatility=ann_v,
                sharpe=sharpe, sortino=sortino, max_dd=mdd, calmar=calmar,
                cvar=cvar, win_rate=win, ir=ir, bear_2022=bear_ret)


# ── Run all three strategies ──────────────────────────────────────────────────
START, END = '2021-03-01', '2026-04-11'

print(f"\nRunning backtests ({START} to {END})...")

print("  1/3  Value-Hedged Momentum (real B/M + E/P)...")
res_vh = run_backtest(START, END, alpha=0.70, n_stocks=50, use_hedge=True, use_value=True)

print("  2/3  Pure Momentum (no value, no hedge)...")
res_pm = run_backtest(START, END, alpha=1.00, n_stocks=50, use_hedge=False, use_value=False)

print("  3/3  S&P 500 benchmark...")
mkt = get_market_index('2020-01-01', '2026-04-25')
bp  = mkt['Close'] if 'Close' in mkt.columns else mkt.iloc[:,0]
if isinstance(bp, pd.DataFrame): bp = bp.iloc[:,0]
bp  = bp.loc[START:END]
br  = bp.pct_change().dropna()
br.index = pd.to_datetime(br.index)
bv  = (1+br).cumprod() * 1_000_000

# Save results
res_vh.to_csv('results/fundamental/vh_history.csv')
res_pm.to_csv('results/fundamental/pm_history.csv')

# ── Compute metrics ───────────────────────────────────────────────────────────
br_idx = br.copy(); br_idx.index = pd.to_datetime(br_idx.index)
m_vh = metrics(res_vh['returns'].set_axis(pd.to_datetime(res_vh['returns'].index)),
               res_vh['portfolio_value'].set_axis(pd.to_datetime(res_vh['portfolio_value'].index)),
               bench=br_idx)

m_pm = metrics(res_pm['returns'].set_axis(pd.to_datetime(res_pm['returns'].index)),
               res_pm['portfolio_value'].set_axis(pd.to_datetime(res_pm['portfolio_value'].index)),
               bench=br_idx)

m_bk = metrics(br_idx, bv)

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "="*78)
print("COMPARISON: Value-Hedged Momentum vs Pure Momentum vs S&P 500")
print("Period:", START, "to", END, " | Real B/M + E/P fundamentals | 45-day lag")
print("="*78)

pct_fmt = {'Total Return','Annualized Return','Annualized Volatility',
           'Max Drawdown','CVaR (5%)','Win Rate','2022 Bear Market'}
rows = [
    ("Total Return",          m_vh['total'],      m_pm['total'],      m_bk['total']),
    ("Annualized Return",     m_vh['ann_return'],  m_pm['ann_return'], m_bk['ann_return']),
    ("Annualized Volatility", m_vh['volatility'],  m_pm['volatility'], m_bk['volatility']),
    ("Sharpe Ratio",          m_vh['sharpe'],      m_pm['sharpe'],     m_bk['sharpe']),
    ("Sortino Ratio",         m_vh['sortino'],     m_pm['sortino'],    m_bk['sortino']),
    ("Max Drawdown",          m_vh['max_dd'],      m_pm['max_dd'],     m_bk['max_dd']),
    ("Calmar Ratio",          m_vh['calmar'],      m_pm['calmar'],     m_bk['calmar']),
    ("CVaR (5%)",             m_vh['cvar'],        m_pm['cvar'],       m_bk['cvar']),
    ("Win Rate",              m_vh['win_rate'],    m_pm['win_rate'],   m_bk['win_rate']),
    ("Information Ratio",     m_vh['ir'],          m_pm['ir'],         np.nan),
    ("2022 Bear Market",      m_vh['bear_2022'],   m_pm['bear_2022'],  m_bk['bear_2022']),
]

hdr = f"{'Metric':<26} {'Value-Hedged Momentum':>23} {'Pure Momentum':>16} {'S&P 500':>10}"
print(hdr)
print("-"*78)

def fmt(v, pct):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v*100:+.2f}%' if pct else f'{v:+.4f}'

for name, a, b, c in rows:
    pct = name in pct_fmt
    print(f"{name:<26} {fmt(a,pct):>23} {fmt(b,pct):>16} {fmt(c,pct):>10}")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

pv_vh = res_vh['portfolio_value'].set_axis(pd.to_datetime(res_vh['portfolio_value'].index)).dropna()
pv_pm = res_pm['portfolio_value'].set_axis(pd.to_datetime(res_pm['portfolio_value'].index)).dropna()

ax = axes[0]
ax.plot(pv_vh/pv_vh.iloc[0], label='Value-Hedged Momentum (real fundamentals)', color='#1F3864', lw=2)
ax.plot(pv_pm/pv_pm.iloc[0], label='Pure Momentum',                             color='#E07B2A', lw=2, ls='--')
ax.plot(bp/bp.iloc[0],        label='S&P 500',                                  color='#2E8B57', lw=2, ls=':')
ax.set_title('Cumulative Returns: Real Fundamental Value Signal vs Pure Momentum vs S&P 500\n'
             f'(B/M + E/P with 45-day reporting lag, {START} to {END})',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Growth of $1')
ax.legend()
ax.grid(alpha=0.3)

ax2 = axes[1]
for pv, label, col, ls in [
    (pv_vh, 'Value-Hedged', '#1F3864', '-'),
    (pv_pm, 'Pure Momentum', '#E07B2A', '--'),
    (bp,    'S&P 500',       '#2E8B57', ':'),
]:
    pk = pv.expanding().max()
    dd = (pv - pk) / pk
    ax2.fill_between(dd.index, dd, 0, alpha=0.2, color=col)
    ax2.plot(dd, label=label, color=col, ls=ls, lw=1.5)
ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
chart = 'results/fundamental/fundamental_comparison.png'
plt.savefig(chart, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart saved: {chart}")
print("\nDone.")
