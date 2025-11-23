"""
Data Transfer Objects (DTOs) for the trading domain.

This module defines the core data structures used for order execution,
including enums for order sides and types, and the TradeOrder dataclass.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class TradeOrder:
    """
    Trade order data transfer object.
    
    This class represents a trading order with all necessary fields.
    The client_order_id field serves as the Idempotency Key.
    
    Attributes:
        symbol: The trading symbol (e.g., "AAPL")
        side: The order side (BUY or SELL)
        qty: The quantity of shares (optional, mutually exclusive with notional)
        notional: The dollar amount to trade (optional, mutually exclusive with qty)
        order_type: The order type (MARKET or LIMIT)
        limit_price: The limit price (required if order_type is LIMIT)
        client_order_id: Unique identifier for idempotency (optional but recommended)
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Optional[float] = None
    notional: Optional[float] = None
    limit_price: Optional[float] = None
    client_order_id: Optional[str] = None
    
    def validate(self) -> None:
        """
        Validate the trade order.
        
        Raises:
            ValueError: If validation fails:
                - Neither qty nor notional is set
                - Both qty and notional are set
                - limit_price is None when order_type is LIMIT
        """
        # Check that exactly one of qty or notional is set
        if self.qty is None and self.notional is None:
            raise ValueError("Either 'qty' or 'notional' must be set")
        
        if self.qty is not None and self.notional is not None:
            raise ValueError("Cannot set both 'qty' and 'notional'. Choose one.")
        
        # Check that limit_price is set for LIMIT orders
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("'limit_price' is required when 'order_type' is LIMIT")


