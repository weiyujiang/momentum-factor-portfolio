# Value-Hedged Momentum Portfolio

A quantitative research system for backtesting crash-hedged momentum strategies using value-based hedging and dynamic risk monitoring.

## Strategy Overview

This system implements a **value-hedged momentum portfolio** that combines:

1. **12-2-1 Momentum Signals**: 12-month lookback, skipping the most recent month to avoid short-term reversal
2. **Value-Based Hedging**: Tilts toward cheaper stocks (higher B/M and E/P ratios) to reduce crash risk
3. **Dynamic Crash Protection**: Monitors VIX, momentum dispersion, and value spreads to adjust hedge allocation

**Goal**: Improve tail risk profile and Sharpe ratio compared to traditional momentum strategies.

## Features

- ✅ S&P 500 universe with survivorship bias mitigation (extended universe of 600-700 stocks)
- ✅ Batch data download from yfinance with intelligent caching
- ✅ 12-2-1 momentum signal calculation (vectorized pandas operations)
- ✅ Value signals from fundamental data (B/M, E/P ratios)
- ✅ Composite crash risk monitoring (VIX + momentum dispersion + value spread)
- ✅ Dynamic hedge allocation (0-20% cash based on risk score)
- ✅ Transaction cost modeling (commission + slippage + market impact ≈ 20 bps per trade)
- ✅ Comprehensive performance analytics (Sharpe, Sortino, max drawdown, tail risk, crash period analysis)
- ✅ Visualization (cumulative returns, drawdown charts, rolling Sharpe)

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Navigate to project directory
cd C:\Users\iamma\value_momentum_portfolio

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the full backtest (2000-2025):

```bash
python main.py
```

This will:
1. Download S&P 500 constituent data
2. Download price and market data (VIX, S&P 500)
3. Calculate momentum and value signals
4. Run monthly rebalancing with dynamic hedging
5. Generate performance metrics and visualizations
6. Save results to `results/` directory

### Output Files

After running, you'll find:

```
results/
├── backtest_results/
│   ├── portfolio_history.csv          # Daily portfolio values and returns
│   ├── performance_metrics.csv        # Summary statistics
│   └── crash_period_analysis.csv      # Performance during major crashes
│
├── figures/
│   ├── cumulative_returns.png         # Portfolio vs S&P 500
│   ├── drawdown.png                   # Drawdown chart
│   └── rolling_sharpe.png             # Rolling 1-year Sharpe ratio
│
└── backtest.log                       # Execution log
```

### Configuration

Modify parameters in [`config/parameters.py`](config/parameters.py):

```python
# Backtest period
START_DATE = '2000-01-01'
END_DATE = '2025-12-31'

# Momentum parameters
MOMENTUM_LOOKBACK = 252  # 12 months
ALPHA = 0.7  # 70% momentum, 30% value

# Portfolio construction
LONG_POSITIONS = 50  # Number of stocks
WEIGHT_METHOD = 'market_cap'  # Weighting scheme

# Risk management
VIX_THRESHOLD = 25.0  # Elevated risk threshold
HEDGE_ALLOCATION_MAX = 0.20  # Max 20% cash hedge

# Transaction costs
COMMISSION = 0.0005  # 5 bps
SLIPPAGE = 0.0010    # 10 bps
MARKET_IMPACT = 0.0005  # 5 bps
```

## Project Structure

```
value_momentum_portfolio/
├── config/                 # Configuration parameters
├── data/                   # Data acquisition modules
│   ├── sp500_constituents.py
│   ├── price_data.py
│   ├── fundamental_data.py
│   └── market_data.py
├── signals/                # Signal generation
│   ├── momentum.py
│   ├── value.py
│   └── combined.py
├── risk/                   # Risk monitoring
│   ├── crash_indicators.py
│   └── risk_metrics.py
├── portfolio/              # Portfolio construction
│   ├── construction.py
│   ├── rebalancing.py
│   └── hedging.py
├── backtest/               # Backtesting engine
│   ├── engine.py
│   ├── transactions.py
│   └── performance.py
├── analytics/              # Performance analytics
│   ├── metrics.py
│   └── visualization.py
├── utils/                  # Utilities
│   ├── cache.py
│   ├── logging_config.py
│   └── helpers.py
├── cache/                  # Cached data
├── results/                # Output directory
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Methodology

### Signal Generation

1. **Momentum (12-2-1)**:
   ```
   MOM = (Price[t-21] / Price[t-273]) - 1
   ```
   - Lookback: 252 days (12 months)
   - Skip: 21 days (1 month) to avoid reversal
   - Captures medium-term price trends

2. **Value (Composite)**:
   ```
   VALUE = Z-score(B/M) + Z-score(E/P)
   ```
   - B/M: Book-to-Market ratio
   - E/P: Earnings-to-Price ratio
   - Higher value = cheaper stock

3. **Combined Signal**:
   ```
   SCORE = 0.7 × rank(MOM) + 0.3 × rank(VALUE)
   ```
   - Select top 50 stocks by combined score
   - Value tilt reduces exposure to expensive momentum stocks

### Crash Risk Monitoring

**Composite Risk Score**:
```
RISK = 0.4×VIX_signal + 0.3×Dispersion_signal + 0.3×Spread_signal
```

Where:
- **VIX Signal**: Normalized VIX level (threshold = 25)
- **Dispersion Signal**: Cross-sectional momentum dispersion (high dispersion = regime instability)
- **Value Spread**: Gap between expensive and cheap stocks (high spread = bubble)

**Hedge Allocation**:
```
HEDGE = min(RISK × 0.20, 0.20)
```
- Linear scaling from 0% to 20% cash based on risk score
- Reduces equity exposure during high-risk periods

### Portfolio Construction

1. **Weight Calculation**: Market-cap weighted within selected stocks
2. **Constraint Application**: Max 5% per position
3. **Hedge Application**: Scale weights by (1 - hedge_allocation)
4. **Rebalancing**: Monthly at month-end
5. **Transaction Costs**: ~20 bps per trade (commission + slippage + impact)

## Known Limitations

### Data Limitations

1. **Survivorship Bias**: Not fully eliminated
   - **Mitigation**: Extended universe of 600-700 stocks, dynamic filtering by market cap
   - **Impact**: Results may be slightly optimistic

2. **Fundamental Data**: Point-in-time approximation
   - **Issue**: yfinance provides current fundamentals, not historical
   - **Mitigation**: Quarterly snapshots with forward-fill
   - **Impact**: Some look-ahead bias in value signals

3. **S&P 500 Constituents**: Approximated via market cap
   - **Issue**: Perfect historical constituent data unavailable
   - **Mitigation**: Dynamic top-500 selection by market cap
   - **Impact**: Slight deviation from actual S&P 500 membership

### Model Limitations

1. **Transaction Costs**: Modeled, not actual execution
2. **Hedge**: Simplified as cash allocation (not actual put options)
3. **Slippage**: Constant model (doesn't account for market conditions)
4. **Market Impact**: Simplified (doesn't scale with trade size)

## Performance Expectations

Based on academic research, a well-implemented value-hedged momentum strategy should achieve:

- **Sharpe Ratio**: 0.7-1.2 (vs S&P 500's ~0.4)
- **Max Drawdown**: 25-40% (vs S&P 500's ~55% in 2008)
- **Annual Return**: 10-15% (after costs)
- **Volatility**: 12-18% annualized
- **Crash Protection**: Positive alpha during 2008, 2020 crashes

## Testing Strategy

### Quick Test (2020-2025)

Modify `config/parameters.py`:
```python
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
```

Then run:
```bash
python main.py
```

This provides faster validation before running the full 25-year backtest.

## Troubleshooting

### Issue: Download failures

**Solution**: Check internet connection, reduce `BATCH_SIZE` in `config/parameters.py`:
```python
BATCH_SIZE = 50  # Reduce from 100 if experiencing failures
```

### Issue: Missing fundamental data

**Solution**: This is expected for some stocks. The system handles missing data gracefully.

### Issue: Memory errors

**Solution**: Reduce universe size or backtest period:
```python
BASE_UNIVERSE_SIZE = 500  # Reduce from 700
```

## Future Enhancements

1. **Better Historical Data**: Incorporate point-in-time fundamental data from paid sources
2. **Options-Based Hedging**: Replace cash hedge with actual put options
3. **Additional Factors**: Quality, profitability, low volatility
4. **Machine Learning**: Optimize signal combination weights
5. **Multi-Asset**: Extend to international equities, commodities
6. **Walk-Forward Optimization**: Dynamic parameter selection

## References

- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Asness, Moskowitz & Pedersen (2013): "Value and Momentum Everywhere"
- Daniel & Moskowitz (2016): "Momentum Crashes"
- Barroso & Santa-Clara (2015): "Momentum Has Its Moments"

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please refer to the documentation or create an issue.

---

**Disclaimer**: Past performance does not guarantee future results. This system is for research purposes only and should not be considered investment advice.
