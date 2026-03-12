"""
Transaction cost modeling.
"""


def calculate_commission(trade_value: float, commission_rate: float = 0.0005) -> float:
    """
    Commission cost calculation.

    Args:
        trade_value: Dollar value of trade
        commission_rate: Commission rate (default 5 bps)

    Returns:
        Commission cost in dollars
    """
    return abs(trade_value) * commission_rate


def calculate_slippage(trade_value: float,
                      volatility: float = 0.02,
                      slippage_rate: float = 0.0010) -> float:
    """
    Slippage cost (price impact from timing).

    Args:
        trade_value: Dollar value of trade
        volatility: Stock volatility (not currently used in simple model)
        slippage_rate: Slippage rate (default 10 bps)

    Returns:
        Slippage cost in dollars
    """
    return abs(trade_value) * slippage_rate


def calculate_market_impact(trade_value: float,
                           daily_volume: float = 1e7,
                           impact_rate: float = 0.0005) -> float:
    """
    Market impact cost (larger trades move price).

    Args:
        trade_value: Dollar value of trade
        daily_volume: Average daily trading volume
        impact_rate: Market impact rate (default 5 bps)

    Returns:
        Market impact cost in dollars
    """
    return abs(trade_value) * impact_rate


def total_transaction_cost(trade_value: float,
                          volatility: float = 0.02,
                          daily_volume: float = 1e7,
                          commission: float = 0.0005,
                          slippage: float = 0.0010,
                          impact: float = 0.0005) -> float:
    """
    Total cost combining all factors.

    Args:
        trade_value: Dollar value of trade
        volatility: Stock volatility
        daily_volume: Daily trading volume
        commission: Commission rate
        slippage: Slippage rate
        impact: Market impact rate

    Returns:
        Total transaction cost in dollars
    """
    total = (
        calculate_commission(trade_value, commission) +
        calculate_slippage(trade_value, volatility, slippage) +
        calculate_market_impact(trade_value, daily_volume, impact)
    )

    return total
