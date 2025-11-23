"""
Execution Engine Module

This module provides broker-agnostic trading execution capabilities
using the Adapter Pattern.
"""

from .engine import TradingEngine
from .models import OrderSide, OrderType, TradeOrder

__all__ = ["TradingEngine", "OrderSide", "OrderType", "TradeOrder"]


