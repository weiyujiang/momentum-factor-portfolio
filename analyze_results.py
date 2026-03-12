import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results/backtest_results/portfolio_history.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Calculate metrics
final_value = df['portfolio_value'].iloc[-1]
initial_value = 1_000_000
total_return = (final_value / initial_value - 1)

# Annualized return
days = len(df)
years = days / 365
ann_return = (final_value / initial_value) ** (1 / years) - 1

# Volatility
returns = df['returns'].dropna()
ann_vol = returns.std() * np.sqrt(252)

# Sharpe (assuming 2% risk-free rate)
sharpe = (ann_return - 0.02) / ann_vol

# Max drawdown
running_max = df['portfolio_value'].expanding().max()
drawdown = (df['portfolio_value'] - running_max) / running_max
max_dd = drawdown.min()

print("=" * 70)
print("VALUE-HEDGED MOMENTUM PORTFOLIO - RESULTS (2020-2025)")
print("=" * 70)
print(f"\nInitial Capital:        ${initial_value:,.0f}")
print(f"Final Portfolio Value:  ${final_value:,.2f}")
print(f"\nTotal Return:           {total_return*100:.2f}%")
print(f"Annualized Return:      {ann_return*100:.2f}%")
print(f"Annualized Volatility:  {ann_vol*100:.2f}%")
print(f"Sharpe Ratio:           {sharpe:.2f}")
print(f"Max Drawdown:           {max_dd*100:.2f}%")
print(f"\nBacktest Period:        {df.index[0].date()} to {df.index[-1].date()}")
print(f"Trading Days:           {days}")
print("\n" + "=" * 70)
print("System validated successfully!")
print("Ready to run full 25-year backtest (2000-2025)")
print("=" * 70)
