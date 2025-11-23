"""
Trading Engine - Main service logic for order execution.

This module provides the TradingEngine class that orchestrates order execution
using the Factory Pattern to select the appropriate broker provider.
"""

import uuid
from typing import Optional

from .interface import ExecutionProvider
from .models import OrderSide, OrderType, PortfolioState, TradeOrder
from .providers.alpaca import AlpacaProvider


class TradingEngine:
    """
    Main trading engine that provides a high-level interface for order execution.
    
    Uses the Factory Pattern to instantiate the appropriate execution provider
    based on the provider name.
    """
    
    def __init__(self, provider_name: str = "alpaca"):
        """
        Initialize the TradingEngine with a specific provider.
        
        Args:
            provider_name: Name of the provider to use (default: "alpaca")
            
        Raises:
            ValueError: If provider_name is not supported
        """
        self.provider = self._create_provider(provider_name)
        
        # Connect to the provider
        if not self.provider.connect():
            raise RuntimeError(f"Failed to connect to {provider_name} provider")
    
    def _create_provider(self, provider_name: str) -> ExecutionProvider:
        """
        Factory method to create the appropriate execution provider.
        
        Args:
            provider_name: Name of the provider to create
            
        Returns:
            ExecutionProvider: An instance of the requested provider
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name.lower() == "alpaca":
            return AlpacaProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get the current portfolio state including cash, total equity, and positions.
        
        Returns:
            PortfolioState: Validated Pydantic model containing:
                - cash: Available cash balance
                - total_equity: Total portfolio value
                - positions: Dictionary mapping symbols to quantities
                - timestamp: When the snapshot was taken
        """
        try:
            return self.provider.get_state()
        except Exception as e:
            print(f"✗ Error getting portfolio state: {str(e)}")
            raise
    
    def place_buy_order(self, symbol: str, amount_usd: float) -> dict:
        """
        Place a market buy order for a specified dollar amount.
        
        Args:
            symbol: The trading symbol (e.g., "AAPL")
            amount_usd: The dollar amount to invest
            
        Returns:
            dict: Raw response from the broker API
            
        Raises:
            ValueError: If order validation fails
            Exception: If order execution fails
        """
        # Generate a unique client_order_id for idempotency
        client_order_id = f"buy_{symbol}_{uuid.uuid4().hex[:8]}"
        
        # Create the trade order
        order = TradeOrder(
            symbol=symbol,
            side=OrderSide.BUY,
            notional=amount_usd,
            order_type=OrderType.MARKET,
            client_order_id=client_order_id
        )
        
        # Execute the order
        return self.provider.execute_order(order)
    
    def close_position(self, symbol: str) -> dict:
        """
        Close an existing position by placing a market sell order.
        
        Args:
            symbol: The trading symbol to close (e.g., "AAPL")
            
        Returns:
            dict: Raw response from the broker API
            
        Raises:
            ValueError: If no position exists for the symbol
            Exception: If order execution fails
        """
        # Get current portfolio state
        state = self.provider.get_state()
        
        # Check if position exists
        if symbol not in state.positions:
            raise ValueError(f"No position found for symbol: {symbol}")
        
        current_qty = state.positions[symbol]
        
        if current_qty == 0:
            raise ValueError(f"Position quantity is zero for symbol: {symbol}")
        
        # Determine the side based on position sign
        # Positive qty = long position (sell to close)
        # Negative qty = short position (buy to close)
        if current_qty > 0:
            side = OrderSide.SELL
            qty = current_qty
        else:
            side = OrderSide.BUY
            qty = abs(current_qty)  # Use absolute value for quantity
        
        # Generate a unique client_order_id for idempotency
        client_order_id = f"close_{symbol}_{uuid.uuid4().hex[:8]}"
        
        # Create the trade order
        order = TradeOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            client_order_id=client_order_id
        )
        
        # Execute the order
        return self.provider.execute_order(order)


