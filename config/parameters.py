"""
Centralized configuration parameters for value-hedged momentum portfolio backtest.
"""

from dataclasses import dataclass
import os


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters"""
    START_DATE: str = '2000-01-01'
    END_DATE: str = '2025-12-31'
    REBALANCE_FREQUENCY: str = 'M'  # 'D', 'W', 'M', 'Q'
    INITIAL_CAPITAL: float = 1_000_000  # $1M starting capital

    # Allow environment variable overrides
    def __post_init__(self):
        self.START_DATE = os.getenv('BACKTEST_START', self.START_DATE)
        self.END_DATE = os.getenv('BACKTEST_END', self.END_DATE)


@dataclass
class SignalConfig:
    """Signal calculation parameters"""
    # Momentum parameters (12-2-1 formation)
    MOMENTUM_LOOKBACK: int = 252      # 12 months (252 trading days)
    MOMENTUM_SKIP: int = 21           # Skip most recent month (21 trading days)
    MOMENTUM_FORMATION: int = 21      # Formation period (21 trading days)

    # Value parameters
    VALUE_PERCENTILE_CUTOFF: int = 50  # Top 50% value stocks

    # Combined signal parameters
    ALPHA: float = 0.7  # Weight on momentum (0.7 = 70% momentum, 30% value)


@dataclass
class PortfolioConfig:
    """Portfolio construction parameters"""
    LONG_POSITIONS: int = 50          # Number of long positions
    WEIGHT_METHOD: str = 'market_cap'  # 'equal', 'market_cap', 'signal', 'risk_parity'
    MAX_POSITION_SIZE: float = 0.05    # 5% max per position
    MIN_POSITION_SIZE: float = 0.001   # 0.1% min per position
    MIN_LIQUIDITY: float = 1e6         # Minimum $1M daily volume
    MAX_TURNOVER: float = 1.0          # Max 100% turnover per rebalance


@dataclass
class RiskConfig:
    """Crash risk monitoring parameters"""
    VIX_THRESHOLD: float = 25.0           # VIX level indicating elevated risk
    DISPERSION_THRESHOLD: float = 0.3     # Momentum dispersion threshold
    VALUE_SPREAD_THRESHOLD: float = 2.0   # Value spread Z-score threshold
    HEDGE_ALLOCATION_MAX: float = 0.20    # Maximum 20% hedge allocation

    # Composite risk weights
    VIX_WEIGHT: float = 0.4               # 40% weight on VIX signal
    DISPERSION_WEIGHT: float = 0.3        # 30% weight on dispersion
    SPREAD_WEIGHT: float = 0.3            # 30% weight on value spread


@dataclass
class CostConfig:
    """Transaction cost parameters"""
    COMMISSION: float = 0.0005           # 5 bps commission per trade
    SLIPPAGE: float = 0.0010             # 10 bps slippage
    MARKET_IMPACT: float = 0.0005        # 5 bps market impact

    @property
    def total_cost_per_trade(self) -> float:
        """Total cost per trade (≈20 bps)"""
        return self.COMMISSION + self.SLIPPAGE + self.MARKET_IMPACT


@dataclass
class UniverseConfig:
    """Universe construction parameters"""
    BASE_UNIVERSE_SIZE: int = 700        # Extended universe size (to reduce survivorship bias)
    ACTIVE_UNIVERSE_SIZE: int = 500      # Active universe size at each rebalance (approx S&P 500)
    MIN_PRICE: float = 5.0               # Minimum stock price ($5 to avoid penny stocks)
    MIN_HISTORY_DAYS: int = 252          # Minimum 1 year of history required


@dataclass
class CacheConfig:
    """Caching configuration"""
    CACHE_DIR: str = 'cache'
    PRICE_CACHE_DIR: str = 'cache/prices'
    FUNDAMENTAL_CACHE_DIR: str = 'cache/fundamentals'
    CONSTITUENT_CACHE_DIR: str = 'cache/constituents'
    CACHE_MAX_AGE_DAYS: int = 7          # Refresh cache if older than 7 days


@dataclass
class DataConfig:
    """Data download configuration"""
    BATCH_SIZE: int = 100                 # Download 100 tickers at a time
    API_DELAY_SECONDS: float = 1.0        # Delay between API calls (rate limiting)
    MAX_RETRIES: int = 3                  # Max retries for failed downloads
    TIMEOUT_SECONDS: int = 30             # Timeout for each download
    MAX_FFILL_DAYS: int = 5               # Max forward-fill for missing data


# Crash periods for analysis
CRASH_PERIODS = {
    'Dot-com Bust': ('2000-03-24', '2002-10-09'),
    'Financial Crisis': ('2007-10-09', '2009-03-09'),
    'Flash Crash': ('2010-05-06', '2010-05-07'),
    'COVID Crash': ('2020-02-19', '2020-03-23'),
    '2022 Bear Market': ('2022-01-03', '2022-10-13'),
}


# Market tickers
MARKET_TICKERS = {
    'sp500': '^GSPC',      # S&P 500 Index
    'vix': '^VIX',         # VIX Volatility Index
    'treasury_10y': '^TNX', # 10-Year Treasury Yield
    'treasury_3m': '^IRX',  # 13-Week Treasury Yield
}


# Risk-free rate (annual)
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate


# Create singleton instances for easy import
backtest_config = BacktestConfig()
signal_config = SignalConfig()
portfolio_config = PortfolioConfig()
risk_config = RiskConfig()
cost_config = CostConfig()
universe_config = UniverseConfig()
cache_config = CacheConfig()
data_config = DataConfig()
