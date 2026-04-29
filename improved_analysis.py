"""
Improved research design:
  1. Recalibrate crash-risk (rolling-percentile dispersion threshold)
  2. Walk-forward validation  (train 2021-2022 / test 2023-2025)
  3. Fama-French 4-factor alpha regression
  4. Parameter sensitivity grid  (alpha x n_stocks)
  5. Block-bootstrap Sharpe confidence intervals
  6. Consolidated comparison chart
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['BACKTEST_START'] = '2020-01-01'
os.environ['BACKTEST_END']   = '2025-03-11'
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import urllib.request, zipfile, io
from scipy import stats

from config.parameters import (backtest_config, portfolio_config,
                               risk_config, MARKET_TICKERS)
from data.sp500_constituents import get_sp500_universe
from data.price_data import PriceDataManager
from data.market_data import get_market_index
from signals.momentum import calculate_momentum_1221, rank_momentum, calculate_momentum_dispersion
from portfolio.construction import PortfolioConstructor
from portfolio.rebalancing import Rebalancer
from backtest.performance import PerformanceTracker
from utils.helpers import get_month_end_dates
from utils.logging_config import setup_logging

setup_logging(log_level='WARNING', log_file='results/improved.log')

os.makedirs('results/improved', exist_ok=True)

# ── Load cached data ──────────────────────────────────────────────────────────
print("Loading cached price & VIX data...")
pm       = PriceDataManager()
universe = get_sp500_universe(use_cache=True, extend=True)
prices   = pm.get_adjusted_prices(universe, '2020-01-01', '2025-03-11')
mom_all  = calculate_momentum_1221(prices)

with open('cache/vix_data.pkl', 'rb') as f:
    vix_data = pickle.load(f)

market_raw = get_market_index('2020-01-01', '2025-03-11')
bench_col  = market_raw['Close'] if 'Close' in market_raw.columns else market_raw.iloc[:, 0]
if isinstance(bench_col, pd.DataFrame):
    bench_col = bench_col.iloc[:, 0]
bench_prices = bench_col
bench_ret    = bench_prices.pct_change().dropna()
bench_ret    = bench_ret[~bench_ret.index.duplicated()]

print(f"Prices: {prices.shape} | Momentum dates: {len(mom_all)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE BACKTEST ENGINE  (recalibrated crash risk + configurable params)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dispersion_threshold(mom_df, lookback_dates, pct=75):
    """Derive dispersion threshold from historical data (rolling-pct calibration)."""
    disps = []
    for d in lookback_dates:
        ds = d.strftime('%Y-%m-%d')
        val = calculate_momentum_dispersion(mom_df, ds)
        if not np.isnan(val):
            disps.append(val)
    if not disps:
        return 0.30
    return np.percentile(disps, pct)


def run_backtest(start_date, end_date,
                 alpha=0.70, n_stocks=50,
                 disp_threshold=None,    # None → use recalibrated value
                 vix_threshold=25.0,
                 use_hedge=True):
    """
    Run backtest and return daily portfolio_value / returns Series.
    """
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

        # Select top stocks by momentum (no value filter — kept simple for grid)
        top_n = min(n_stocks * 2, len(mom_scores))
        candidates = mom_scores.nlargest(top_n)

        # Value tilt: exclude top-decile extreme scorers (simplified overlay)
        if alpha < 1.0:
            cutoff = candidates.quantile(0.9)
            candidates = candidates[candidates < cutoff]

        selected = candidates.head(n_stocks).index.tolist()
        if not selected:
            continue

        target_weights = pc.construct_long_portfolio(
            selected_tickers=selected,
            fundamentals=None,
            prices=prices,
            signals=mom_scores,
        )

        # ── Recalibrated crash-risk hedge ─────────────────────────────────
        hedge_alloc = 0.0
        if use_hedge:
            vix_val = float(vix_data.loc[date_str, 'VIX']) if date_str in vix_data.index else 25.0
            vix_sig = max(0.0, (vix_val - vix_threshold) / vix_threshold)

            # Dispersion signal with calibrated threshold
            disp_thr = disp_threshold if disp_threshold is not None else 0.30
            disp_raw = calculate_momentum_dispersion(mom_all, date_str)
            disp_sig = 0.0 if np.isnan(disp_raw) else max(0.0, (disp_raw - disp_thr) / disp_thr)

            composite   = 0.4 * vix_sig + 0.3 * disp_sig   # value spread = 0 (no fundamentals)
            hedge_alloc = min(composite * risk_config.HEDGE_ALLOCATION_MAX,
                              risk_config.HEDGE_ALLOCATION_MAX)

            target_weights = pc.apply_hedge(target_weights, hedge_alloc)

        new_weights, costs = reb.execute_rebalance(curr_weights, target_weights, port_value)
        curr_weights = new_weights
        port_value  -= costs
        perf.record_rebalance(date_str, 0, costs, hedge_alloc)

        # Mark to market until next rebalance
        next_date = (rebal_dates[i + 1] if i < len(rebal_dates) - 1
                     else pd.Timestamp(end_date))
        for dt in prices.loc[rebal_date:next_date].index:
            if len(curr_weights) == 0:
                continue
            day_r = prices.shift(1).loc[dt]
            r     = ((prices.loc[dt] - day_r) / day_r).reindex(curr_weights.index).fillna(0)
            pr    = float((curr_weights * r).sum())
            port_value *= (1 + pr)
            ds = dt.strftime('%Y-%m-%d')
            perf.record_daily_value(ds, port_value)
            perf.record_daily_return(ds, pr)

    vals = perf.get_value_series()
    rets = perf.get_returns_series()
    return pd.DataFrame({'portfolio_value': vals, 'returns': rets})


# ── Metrics helper ────────────────────────────────────────────────────────────
def metrics(rets, vals, rf=0.02):
    rets = rets.dropna()
    if len(rets) < 20:
        return {}
    n = len(rets)
    total  = (1 + rets).prod() - 1
    ann_r  = (1 + total) ** (252 / n) - 1
    ann_v  = rets.std() * np.sqrt(252)
    sharpe = (ann_r - rf) / ann_v if ann_v > 0 else np.nan
    ds_ret = rets[rets < 0]
    sortino = (ann_r - rf) / (ds_ret.std() * np.sqrt(252)) if len(ds_ret) > 0 else np.nan
    peak   = vals.expanding().max()
    mdd    = float(((vals - peak) / peak).min())
    calmar = ann_r / abs(mdd) if mdd != 0 else np.nan
    cvar   = float(-rets[rets <= rets.quantile(0.05)].mean())
    return dict(ann_return=ann_r, volatility=ann_v, sharpe=sharpe,
                sortino=sortino, max_dd=mdd, calmar=calmar, cvar=cvar,
                total_return=total, win_rate=float((rets > 0).mean()))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RECALIBRATED CRASH RISK  (rolling-percentile dispersion threshold)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("1. RECALIBRATED CRASH RISK")
print("="*65)

# Calibrate dispersion threshold from 2021 training data
train_dates = [d for d in mom_all.index if '2021-03-01' <= d.strftime('%Y-%m-%d') <= '2022-12-31']
disp_thr_calibrated = compute_dispersion_threshold(mom_all, train_dates, pct=75)
print(f"Calibrated dispersion threshold (75th pct of 2021-2022): {disp_thr_calibrated:.4f}")
print(f"Original threshold: 0.3000  ->  ratio: {disp_thr_calibrated/0.3:.1f}x higher")

# Original (miscalibrated) vs recalibrated
print("\nRunning original strategy (miscalibrated)...")
orig = run_backtest('2021-03-01', '2025-03-11', use_hedge=True, disp_threshold=0.30)

print("Running recalibrated strategy...")
recal = run_backtest('2021-03-01', '2025-03-11', use_hedge=True, disp_threshold=disp_thr_calibrated)

print("Running no-hedge baseline...")
nohd = run_backtest('2021-03-01', '2025-03-11', use_hedge=False)

m_orig  = metrics(orig['returns'],  orig['portfolio_value'])
m_recal = metrics(recal['returns'], recal['portfolio_value'])
m_nohd  = metrics(nohd['returns'],  nohd['portfolio_value'])

print(f"\n{'Metric':<22} {'Original':>12} {'Recalibrated':>14} {'No Hedge':>12}")
print("-"*62)
for k in ['ann_return','volatility','sharpe','max_dd','total_return']:
    v1, v2, v3 = m_orig.get(k,np.nan), m_recal.get(k,np.nan), m_nohd.get(k,np.nan)
    print(f"{k:<22} {v1:>12.4f} {v2:>14.4f} {v3:>12.4f}")

orig.to_csv('results/improved/orig_history.csv')
recal.to_csv('results/improved/recal_history.csv')
nohd.to_csv('results/improved/nohd_history.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("2. WALK-FORWARD VALIDATION  (train 2021-2022 / test 2023-2025)")
print("="*65)

TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END   = '2025-03-11'

# Calibrate threshold on training data only
train_idx = [d for d in mom_all.index
             if '2021-03-01' <= d.strftime('%Y-%m-%d') <= TRAIN_END]
disp_thr_wf = compute_dispersion_threshold(mom_all, train_idx, pct=75)
print(f"Walk-forward dispersion threshold (from train): {disp_thr_wf:.4f}")

# In-sample (training period)
print("In-sample (2021-2022)...")
wf_train = run_backtest('2021-03-01', TRAIN_END, disp_threshold=disp_thr_wf)
m_train  = metrics(wf_train['returns'], wf_train['portfolio_value'])

# Out-of-sample (test period — parameters locked from training)
print("Out-of-sample (2023-2025)...")
wf_test  = run_backtest(TEST_START, TEST_END, disp_threshold=disp_thr_wf)
m_test   = metrics(wf_test['returns'], wf_test['portfolio_value'])

# Benchmark splits
bench_train = bench_ret.loc['2021-03-01':TRAIN_END].dropna()
bench_test  = bench_ret.loc[TEST_START:TEST_END].dropna()
bv_train    = (1 + bench_train).cumprod() * 1_000_000
bv_test     = (1 + bench_test).cumprod()  * 1_000_000
m_btrain    = metrics(bench_train, bv_train)
m_btest     = metrics(bench_test,  bv_test)

print(f"\n{'Metric':<22} {'In-Sample':>12} {'Out-of-Sample':>15} {'Bench(IS)':>12} {'Bench(OOS)':>12}")
print("-"*75)
for k in ['ann_return','volatility','sharpe','max_dd']:
    v1 = m_train.get(k, np.nan); v2 = m_test.get(k, np.nan)
    v3 = m_btrain.get(k,np.nan); v4 = m_btest.get(k,np.nan)
    print(f"{k:<22} {v1:>12.4f} {v2:>15.4f} {v3:>12.4f} {v4:>12.4f}")

wf_train.to_csv('results/improved/wf_train.csv')
wf_test.to_csv('results/improved/wf_test.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAMA-FRENCH 4-FACTOR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("3. FAMA-FRENCH 4-FACTOR ALPHA REGRESSION")
print("="*65)

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

def fetch_ff(url, skip=3):
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            name = [n for n in zf.namelist() if n.endswith('.CSV')][0]
            raw  = zf.read(name).decode('latin-1')
        lines = raw.splitlines()
        # Find data start (first line that starts with a digit after skip lines)
        start = 0
        for i, l in enumerate(lines):
            if i > skip and l.strip() and l.strip()[0].isdigit():
                start = i
                break
        # Find data end (blank line or "Annual" section)
        end = len(lines)
        for i in range(start, len(lines)):
            stripped = lines[i].strip()
            if not stripped or (stripped and not stripped[0].isdigit() and i > start + 5):
                end = i
                break
        data = '\n'.join(lines[start:end])
        df = pd.read_csv(io.StringIO(data), header=None)
        return df
    except Exception as e:
        print(f"  FF download failed: {e}")
        return None

print("Downloading Fama-French factors...")
ff3  = fetch_ff(FF_URL)
mom_f = fetch_ff(MOM_URL, skip=13)

ff_ready = False
if ff3 is not None and mom_f is not None:
    try:
        ff3.columns  = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
        ff3['Date']  = pd.to_datetime(ff3['Date'].astype(str).str.strip(), format='%Y%m%d', errors='coerce')
        ff3 = ff3.dropna(subset=['Date']).set_index('Date')
        for c in ff3.columns:
            ff3[c] = pd.to_numeric(ff3[c], errors='coerce') / 100

        mom_f.columns = ['Date', 'Mom']
        mom_f['Date'] = pd.to_datetime(mom_f['Date'].astype(str).str.strip(), format='%Y%m%d', errors='coerce')
        mom_f = mom_f.dropna(subset=['Date']).set_index('Date')
        mom_f['Mom'] = pd.to_numeric(mom_f['Mom'], errors='coerce') / 100

        ff4 = ff3.join(mom_f, how='inner')
        ff4 = ff4.loc['2021-03-01':'2025-03-11'].dropna()
        print(f"  FF4 factors: {len(ff4)} days")
        ff_ready = True
    except Exception as e:
        print(f"  FF parsing failed: {e}")

def ff_regression(port_rets, ff4, label):
    common = port_rets.index.intersection(ff4.index)
    if len(common) < 30:
        print(f"  {label}: insufficient overlap ({len(common)} days)")
        return
    pr  = port_rets.loc[common]
    fac = ff4.loc[common]
    y   = pr - fac['RF']
    X   = fac[['Mkt-RF', 'SMB', 'HML', 'Mom']].copy()
    X.insert(0, 'const', 1.0)
    b, res, _, _ = np.linalg.lstsq(X.values, y.values, rcond=None)
    yhat   = X.values @ b
    ss_res = float(np.sum((y.values - yhat)**2))
    ss_tot = float(np.sum((y.values - y.values.mean())**2))
    r2     = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    n, k   = len(y), X.shape[1]
    se     = np.sqrt(np.diag(np.linalg.inv(X.values.T @ X.values) * ss_res / (n-k)))
    t_alpha = b[0] / se[0]
    ann_alpha = b[0] * 252
    p_val  = 2 * (1 - stats.t.cdf(abs(t_alpha), df=n-k))
    stars  = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
    print(f"\n  {label}")
    print(f"    Ann. Alpha:  {ann_alpha:+.4f}  (t={t_alpha:+.2f}, p={p_val:.3f}) {stars}")
    print(f"    Mkt-RF beta: {b[1]:.3f}   SMB: {b[2]:.3f}   HML: {b[3]:.3f}   Mom: {b[4]:.3f}")
    print(f"    R²:          {r2:.3f}")
    return dict(alpha=ann_alpha, t_alpha=t_alpha, p_val=p_val,
                mkt=b[1], smb=b[2], hml=b[3], mom_b=b[4], r2=r2)

if ff_ready:
    r_orig  = orig['returns'].dropna()
    r_recal = recal['returns'].dropna()
    r_nohd  = nohd['returns'].dropna()
    r_orig.index  = pd.to_datetime(r_orig.index)
    r_recal.index = pd.to_datetime(r_recal.index)
    r_nohd.index  = pd.to_datetime(r_nohd.index)
    ff_orig  = ff_regression(r_orig,  ff4, "Original (miscalibrated hedge)")
    ff_recal = ff_regression(r_recal, ff4, "Recalibrated hedge")
    ff_nohd  = ff_regression(r_nohd,  ff4, "No hedge (pure momentum)")
else:
    print("  Skipping FF regression (download failed)")
    ff_orig = ff_recal = ff_nohd = None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PARAMETER SENSITIVITY GRID
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("4. PARAMETER SENSITIVITY GRID  (alpha × n_stocks)")
print("="*65)

alphas   = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_stocks_grid = [20, 30, 50, 75]
grid_results = {}

for a in alphas:
    for n in n_stocks_grid:
        res = run_backtest('2021-03-01', '2025-03-11',
                           alpha=a, n_stocks=n,
                           disp_threshold=disp_thr_calibrated,
                           use_hedge=True)
        m = metrics(res['returns'], res['portfolio_value'])
        grid_results[(a, n)] = m.get('sharpe', np.nan)
        print(f"  alpha={a:.1f}  n={n:3d}  Sharpe={grid_results[(a,n)]:.3f}")

# Build grid table
grid_df = pd.DataFrame(
    {n: {a: grid_results[(a, n)] for a in alphas} for n in n_stocks_grid}
)
grid_df.index.name = 'alpha \\ n_stocks'
print("\nSharpe Ratio Grid:")
print(grid_df.round(3).to_string())
grid_df.to_csv('results/improved/sensitivity_grid.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BLOCK-BOOTSTRAP SHARPE CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("5. BLOCK-BOOTSTRAP SHARPE 95% CONFIDENCE INTERVALS")
print("="*65)

def block_bootstrap_sharpe(rets, n_boot=2000, block_size=21, rf=0.02):
    rets = rets.dropna().values
    n    = len(rets)
    sharpes = []
    for _ in range(n_boot):
        idx   = np.random.randint(0, n - block_size, size=(n // block_size) + 1)
        sample = np.concatenate([rets[i:i+block_size] for i in idx])[:n]
        ann_r  = (1 + sample).prod() ** (252/n) - 1
        ann_v  = sample.std() * np.sqrt(252)
        sharpes.append((ann_r - rf) / ann_v if ann_v > 0 else np.nan)
    sharpes = np.array([s for s in sharpes if not np.isnan(s)])
    return np.percentile(sharpes, [2.5, 50, 97.5])

for label, res in [("Original hedge",    orig),
                   ("Recalibrated hedge", recal),
                   ("No hedge",           nohd)]:
    lo, med, hi = block_bootstrap_sharpe(res['returns'])
    print(f"  {label:<22}  Sharpe 95% CI: [{lo:.3f},  {hi:.3f}]  (median {med:.3f})")

# Benchmark CI
bench_sub = bench_ret.loc['2021-03-01':'2025-03-11'].dropna()
lo, med, hi = block_bootstrap_sharpe(bench_sub)
print(f"  {'S&P 500':<22}  Sharpe 95% CI: [{lo:.3f},  {hi:.3f}]  (median {med:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONSOLIDATED COMPARISON CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("6. GENERATING CHARTS")
print("="*65)

fig = plt.figure(figsize=(16, 20))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.35)

COLORS = {'orig':'#888888', 'recal':'#1F3864', 'nohd':'#E07B2A', 'bench':'#2E8B57'}

def norm(series):
    s = series.dropna()
    return s / s.iloc[0]

# ── Panel A: Cumulative returns (recalibrated comparison) ─────────────────────
ax1 = fig.add_subplot(gs[0, :])
for label, res, col, ls in [
    ("Original hedge (miscalibrated)", orig,  COLORS['orig'],  '--'),
    ("Recalibrated hedge",             recal, COLORS['recal'], '-'),
    ("No hedge (pure momentum)",       nohd,  COLORS['nohd'],  '-.'),
]:
    pv = res['portfolio_value'].dropna()
    pv.index = pd.to_datetime(pv.index)
    ax1.plot(pv / pv.iloc[0], label=label, color=col, ls=ls, lw=1.8)

bv = bench_prices.loc['2021-03-01':'2025-03-11']
ax1.plot(bv / bv.iloc[0], label="S&P 500", color=COLORS['bench'], ls=':', lw=1.8)
ax1.set_title("A.  Cumulative Returns: Original vs. Recalibrated vs. No Hedge (2021–2025)",
              fontsize=12, fontweight='bold')
ax1.set_ylabel("Growth of $1")
ax1.legend(fontsize=9)
ax1.axvline(pd.Timestamp('2023-01-01'), color='red', ls='--', lw=1, alpha=0.5)
ax1.text(pd.Timestamp('2023-01-15'), ax1.get_ylim()[0]*1.01, 'OOS start', color='red', fontsize=8)
ax1.grid(alpha=0.3)

# ── Panel B: Walk-forward IS vs OOS ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
for label, res, col, ls in [
    ("In-Sample (2021-22)",    wf_train, COLORS['recal'], '-'),
    ("Out-of-Sample (2023-25)",wf_test,  COLORS['nohd'],  '--'),
]:
    pv = res['portfolio_value'].dropna()
    pv.index = pd.to_datetime(pv.index)
    ax2.plot(pv / pv.iloc[0], label=label, color=col, ls=ls, lw=1.8)
ax2.set_title("B.  Walk-Forward: In-Sample vs Out-of-Sample", fontsize=11, fontweight='bold')
ax2.set_ylabel("Growth of $1")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ── Panel C: Drawdown ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
for label, res, col, ls in [
    ("Recalibrated hedge", recal, COLORS['recal'], '-'),
    ("No hedge",           nohd,  COLORS['nohd'],  '-.'),
    ("S&P 500",            None,  COLORS['bench'], ':'),
]:
    if res is not None:
        pv = res['portfolio_value'].dropna()
        pv.index = pd.to_datetime(pv.index)
    else:
        pv = bench_prices.loc['2021-03-01':'2025-03-11']
    pk = pv.expanding().max()
    dd = (pv - pk) / pk
    ax3.fill_between(dd.index, dd, 0, alpha=0.25, color=col)
    ax3.plot(dd, label=label, color=col, ls=ls, lw=1.3)
ax3.set_title("C.  Drawdown Comparison", fontsize=11, fontweight='bold')
ax3.set_ylabel("Drawdown")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# ── Panel D: Sharpe Sensitivity Heatmap ──────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
heat = grid_df.values.astype(float)
im   = ax4.imshow(heat, cmap='RdYlGn', aspect='auto',
                  vmin=np.nanmin(heat)-0.05, vmax=np.nanmax(heat)+0.05)
ax4.set_xticks(range(len(n_stocks_grid)))
ax4.set_xticklabels([str(n) for n in n_stocks_grid])
ax4.set_yticks(range(len(alphas)))
ax4.set_yticklabels([str(a) for a in alphas])
ax4.set_xlabel("N stocks")
ax4.set_ylabel("Alpha (momentum weight)")
ax4.set_title("D.  Sharpe Ratio Sensitivity\n(alpha × n_stocks)", fontsize=11, fontweight='bold')
for i in range(len(alphas)):
    for j in range(len(n_stocks_grid)):
        val = heat[i, j]
        ax4.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=9,
                 color='black' if 0.3 < val < 0.8 else 'white')
plt.colorbar(im, ax=ax4, shrink=0.8)

# ── Panel E: Bootstrap Sharpe CIs ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
boot_labels  = ["Original\nhedge", "Recalibrated\nhedge", "No hedge", "S&P 500"]
boot_series  = [orig['returns'], recal['returns'], nohd['returns'], bench_sub]
boot_cols    = [COLORS['orig'], COLORS['recal'], COLORS['nohd'], COLORS['bench']]
boot_x       = np.arange(len(boot_labels))
for i, (ser, col) in enumerate(zip(boot_series, boot_cols)):
    lo, med, hi = block_bootstrap_sharpe(ser)
    ax5.bar(i, med, color=col, alpha=0.8, width=0.5)
    ax5.errorbar(i, med, yerr=[[med-lo],[hi-med]], fmt='none',
                 color='black', capsize=6, lw=2)
ax5.axhline(0, color='black', lw=0.8)
ax5.set_xticks(boot_x)
ax5.set_xticklabels(boot_labels, fontsize=9)
ax5.set_ylabel("Sharpe Ratio")
ax5.set_title("E.  Bootstrap 95% CI for Sharpe Ratio\n(block size = 21 days, 2000 draws)",
              fontsize=11, fontweight='bold')
ax5.grid(alpha=0.3, axis='y')

# ── Panel F: Rolling 6-month Sharpe ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
window = 126
for label, res, col, ls in [
    ("Recalibrated hedge", recal, COLORS['recal'], '-'),
    ("No hedge",           nohd,  COLORS['nohd'],  '-.'),
    ("S&P 500",            None,  COLORS['bench'], ':'),
]:
    r = res['returns'] if res is not None else bench_ret.loc['2021-03-01':'2025-03-11']
    r.index = pd.to_datetime(r.index)
    roll_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)
    ax6.plot(roll_sharpe, label=label, color=col, ls=ls, lw=1.6)
ax6.axhline(0, color='black', lw=0.8)
ax6.axvline(pd.Timestamp('2023-01-01'), color='red', ls='--', lw=1, alpha=0.5)
ax6.set_title("F.  Rolling 6-Month Sharpe Ratio", fontsize=11, fontweight='bold')
ax6.set_ylabel("Sharpe (annualised)")
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)
ax6.fill_betweenx(ax6.get_ylim(), pd.Timestamp('2023-01-01'),
                  pd.Timestamp('2025-03-11'), alpha=0.06, color='red', label='OOS period')

fig.suptitle("Value-Hedged Momentum: Improved Research Design\n"
             "Walk-Forward Validation  •  Recalibrated Crash Risk  •  Sensitivity Analysis",
             fontsize=14, fontweight='bold', y=1.01)

chart_path = 'results/improved/improved_analysis.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved: {chart_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SUMMARY: ALL STRATEGIES (2021-03 to 2025-03)")
print("="*65)

bench_full = bench_ret.loc['2021-03-01':'2025-03-11'].dropna()
bv_full    = (1 + bench_full).cumprod() * 1_000_000
m_bench    = metrics(bench_full, bv_full)

rows = [
    ("S&P 500 Benchmark",          m_bench),
    ("Pure Momentum (no hedge)",   m_nohd),
    ("Original Hedge (miscalib.)", m_orig),
    ("Recalibrated Hedge",         m_recal),
    ("Walk-Fwd IN-SAMPLE",         m_train),
    ("Walk-Fwd OUT-OF-SAMPLE",     m_test),
]

hdr = f"{'Strategy':<30} {'Ann.Ret':>9} {'Vol':>7} {'Sharpe':>8} {'Max DD':>8} {'Calmar':>8}"
print(hdr)
print("-"*72)
for name, m in rows:
    sep = "──" if "OOS" in name else ""
    print(f"{sep+name:<30} {m.get('ann_return',np.nan):>9.4f} "
          f"{m.get('volatility',np.nan):>7.4f} "
          f"{m.get('sharpe',np.nan):>8.4f} "
          f"{m.get('max_dd',np.nan):>8.4f} "
          f"{m.get('calmar',np.nan):>8.4f}")

print("\nDone. Results in results/improved/")
