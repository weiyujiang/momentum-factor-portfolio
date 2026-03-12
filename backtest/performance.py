"""
Performance tracking during backtest.
"""

import pandas as pd
from typing import List, Dict


class PerformanceTracker:
    """Track portfolio performance metrics during backtest."""

    def __init__(self):
        """Initialize PerformanceTracker."""
        self.daily_values = []
        self.daily_returns = []
        self.rebalance_dates = []
        self.turnover_history = []
        self.cost_history = []
        self.hedge_history = []

    def record_daily_value(self, date: str, value: float):
        """Record daily portfolio value."""
        self.daily_values.append({'date': date, 'value': value})

    def record_daily_return(self, date: str, ret: float):
        """Record daily portfolio return."""
        self.daily_returns.append({'date': date, 'return': ret})

    def record_rebalance(self, date: str, turnover: float, cost: float,
                        hedge_allocation: float = 0.0):
        """Record rebalancing event."""
        self.rebalance_dates.append(date)
        self.turnover_history.append({'date': date, 'turnover': turnover})
        self.cost_history.append({'date': date, 'cost': cost})
        self.hedge_history.append({'date': date, 'hedge': hedge_allocation})

    def get_returns_series(self) -> pd.Series:
        """Get daily returns as pandas Series."""
        if not self.daily_returns:
            return pd.Series()

        df = pd.DataFrame(self.daily_returns)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df['return']

    def get_value_series(self) -> pd.Series:
        """Get portfolio values as pandas Series."""
        if not self.daily_values:
            return pd.Series()

        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df['value']

    def get_performance_summary(self) -> Dict:
        """Get summary statistics."""
        returns = self.get_returns_series()
        values = self.get_value_series()

        if len(returns) == 0:
            return {}

        summary = {
            'total_days': len(values),
            'total_rebalances': len(self.rebalance_dates),
            'avg_turnover': pd.DataFrame(self.turnover_history)['turnover'].mean() if self.turnover_history else 0,
            'total_costs': pd.DataFrame(self.cost_history)['cost'].sum() if self.cost_history else 0,
            'avg_hedge': pd.DataFrame(self.hedge_history)['hedge'].mean() if self.hedge_history else 0,
        }

        return summary
