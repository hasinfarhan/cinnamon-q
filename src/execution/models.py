"""
Data Transfer Objects (DTOs) for the trading domain.

This module defines the core data structures used for order execution,
including enums for order sides and types, and Pydantic models for
state management and order execution.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class TradeOrder(BaseModel):
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
    
    @model_validator(mode='after')
    def validate_order(self) -> 'TradeOrder':
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
        
        return self


class TargetPortfolio(BaseModel):
    """
    Represents the desired distribution of assets (the Brain's target state).
    
    This model enforces strict "Halal" constraints:
    - No Leverage: Sum of weights must be <= 1.0 (with epsilon tolerance)
    - Long Only: No negative weights
    
    Attributes:
        timestamp: When this target was created
        weights: Dictionary mapping ticker symbols to target weights (0.0 to 1.0)
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    weights: Dict[str, float] = Field(default_factory=dict)
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """
        Validate that all weights are non-negative (Long Only constraint).
        
        Raises:
            ValueError: If any weight is negative
        """
        for ticker, weight in v.items():
            if weight < 0:
                raise ValueError(
                    f"Negative weight not allowed (Long Only constraint). "
                    f"Ticker '{ticker}' has weight {weight}"
                )
        return v
    
    @model_validator(mode='after')
    def validate_no_leverage(self) -> 'TargetPortfolio':
        """
        Validate that sum of weights does not exceed 1.0 (No Leverage constraint).
        
        Uses epsilon tolerance (1e-6) to handle floating point precision issues.
        
        Raises:
            ValueError: If sum of weights > 1.0 + epsilon
        """
        total_weight = sum(self.weights.values())
        epsilon = 1e-6
        
        if total_weight > 1.0 + epsilon:
            raise ValueError(
                f"No Leverage constraint violated: Sum of weights ({total_weight:.6f}) "
                f"exceeds 1.0 (tolerance: {epsilon})"
            )
        
        return self
    
    model_config = ConfigDict(frozen=True)  # Make model immutable to prevent accidental modification


class PortfolioState(BaseModel):
    """
    Represents the broker's current snapshot of the portfolio.
    
    Attributes:
        timestamp: When this snapshot was taken
        cash: Available cash balance in USD
        total_equity: Total portfolio equity (cash + positions value) in USD
        positions: Dictionary mapping ticker symbols to quantities (shares)
        valuations: Dictionary mapping ticker symbols to current market value in USD
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    cash: float = Field(ge=0.0, description="Available cash balance (must be >= 0)")
    total_equity: float = Field(ge=0.0, description="Total portfolio equity (must be >= 0)")
    positions: Dict[str, float] = Field(default_factory=dict, description="Ticker -> quantity mapping")
    valuations: Dict[str, float] = Field(default_factory=dict, description="Ticker -> market value in USD")
    
    model_config = ConfigDict(frozen=True)  # Make model immutable to prevent accidental modification
