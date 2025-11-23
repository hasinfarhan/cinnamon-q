"""
Execution Engine Module

This module provides broker-agnostic trading execution capabilities
using the Adapter Pattern.
"""

from .engine import TradingEngine
from .models import (
    OrderSide,
    OrderType,
    TradeOrder,
    TargetPortfolio,
    PortfolioState
)

__all__ = [
    "TradingEngine",
    "OrderSide",
    "OrderType",
    "TradeOrder",
    "TargetPortfolio",
    "PortfolioState"
]


