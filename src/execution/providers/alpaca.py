"""
Alpaca Trading API Provider Implementation.

This module implements the ExecutionProvider interface using the Alpaca API.
"""

import os
from datetime import datetime
from typing import Dict

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from ..interface import ExecutionProvider
from ..models import OrderSide, OrderType, PortfolioState, TradeOrder

# Load environment variables
load_dotenv()


class AlpacaProvider(ExecutionProvider):
    """
    Alpaca broker adapter implementation.
    
    This class provides a concrete implementation of the ExecutionProvider
    interface using the Alpaca Trading API.
    """
    
    def __init__(self):
        """Initialize the Alpaca provider with credentials from environment."""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.api = None
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment variables"
            )
    
    def connect(self) -> bool:
        """
        Authenticate and establish connection with Alpaca.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version="v2"
            )
            # Test connection by getting account info
            account = self.api.get_account()
            print(f"✓ Successfully connected to Alpaca. Account status: {account.status}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Alpaca: {str(e)}")
            return False
    
    def get_state(self) -> PortfolioState:
        """
        Get the current portfolio state from Alpaca.
        
        Returns:
            PortfolioState: Validated Pydantic model containing cash, total_equity, and positions
        """
        try:
            if not self.api:
                raise RuntimeError("Not connected to Alpaca. Call connect() first.")
            
            # Get account information
            account = self.api.get_account()
            
            # Alpaca API returns values as strings, so we need explicit type conversion
            cash = float(account.cash)
            total_equity = float(account.portfolio_value)  # portfolio_value is the total equity
            
            # Get positions
            positions = self.api.list_positions()
            position_dict: Dict[str, float] = {}
            valuations_dict: Dict[str, float] = {}
            
            for position in positions:
                symbol = position.symbol
                # Alpaca returns qty as string, convert to float
                qty = float(position.qty)
                position_dict[symbol] = qty
                
                # Extract market_value for valuations
                # Alpaca returns market_value as string, convert to float
                market_value = float(position.market_value)
                valuations_dict[symbol] = market_value
            
            # Create and return the validated PortfolioState model
            # Pydantic will validate that cash and total_equity are >= 0
            return PortfolioState(
                timestamp=datetime.now(),
                cash=cash,
                total_equity=total_equity,
                positions=position_dict,
                valuations=valuations_dict
            )
        except Exception as e:
            print(f"✗ Error getting portfolio state: {str(e)}")
            raise
    
    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions from Alpaca.
        
        Returns:
            Dict[str, float]: Dictionary mapping ticker symbols to quantities.
                Positive values indicate long positions, negative values indicate short positions.
        """
        try:
            if not self.api:
                raise RuntimeError("Not connected to Alpaca. Call connect() first.")
            
            positions = self.api.list_positions()
            position_dict = {}
            
            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                position_dict[symbol] = qty
            
            return position_dict
        except Exception as e:
            print(f"✗ Error getting positions: {str(e)}")
            raise
    
    def get_cash_balance(self) -> float:
        """
        Get the total available cash balance from Alpaca.
        
        Returns:
            float: Available cash balance in USD
        """
        try:
            if not self.api:
                raise RuntimeError("Not connected to Alpaca. Call connect() first.")
            
            account = self.api.get_account()
            cash = float(account.cash)
            return cash
        except Exception as e:
            print(f"✗ Error getting cash balance: {str(e)}")
            raise
    
    def execute_order(self, order: TradeOrder) -> dict:
        """
        Execute a trade order with Alpaca.
        
        Args:
            order: The TradeOrder object containing order details
            
        Returns:
            dict: Raw response from Alpaca API
            
        Raises:
            ValueError: If order validation fails
            Exception: If order execution fails
        """
        try:
            # Order validation happens automatically via Pydantic when TradeOrder is instantiated
            if not self.api:
                raise RuntimeError("Not connected to Alpaca. Call connect() first.")
            
            # Map our TradeOrder to Alpaca's submit_order parameters
            symbol = order.symbol
            side = order.side.value  # Convert enum to string
            order_type = order.order_type.value  # Convert enum to string
            
            # Determine quantity or notional
            qty = None
            notional = None
            if order.qty is not None:
                qty = order.qty
            elif order.notional is not None:
                notional = order.notional
            
            # Build order parameters
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
            }
            
            # Add quantity or notional
            if qty is not None:
                order_params["qty"] = qty
            elif notional is not None:
                order_params["notional"] = notional
            
            # Add limit price if it's a limit order
            if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                order_params["limit_price"] = order.limit_price
            
            # CRITICAL: Add client_order_id for idempotency
            if order.client_order_id:
                order_params["client_order_id"] = order.client_order_id
            
            # FIX: Crypto requires 'gtc', Stocks use 'day'
            # Simple heuristic: Crypto pairs on Alpaca usually contain '/' (e.g. BTC/USD)
            # Stocks are single ticker symbols (e.g. AAPL)
            time_in_force = 'gtc' if '/' in symbol else 'day'
            order_params["time_in_force"] = time_in_force
            
            # Submit the order
            print(f"Submitting order: {symbol} {side} {order_type} (time_in_force: {time_in_force})")
            if qty:
                print(f"  Quantity: {qty}")
            if notional:
                print(f"  Notional: ${notional:.2f}")
            if order.limit_price:
                print(f"  Limit Price: ${order.limit_price:.2f}")
            if order.client_order_id:
                print(f"  Client Order ID: {order.client_order_id}")
            
            response = self.api.submit_order(**order_params)
            
            # Convert response to dict
            result = {
                "id": response.id,
                "client_order_id": response.client_order_id,
                "symbol": response.symbol,
                "side": response.side,
                "order_type": response.order_type,
                "qty": float(response.qty) if hasattr(response, 'qty') and response.qty is not None else None,
                "status": response.status,
                "filled_qty": float(response.filled_qty) if hasattr(response, 'filled_qty') and response.filled_qty is not None else 0.0,
            }
            
            print(f"✓ Order submitted successfully. Order ID: {response.id}, Status: {response.status}")
            return result
            
        except ValueError as e:
            print(f"✗ Order validation error: {str(e)}")
            raise
        except Exception as e:
            print(f"✗ Error executing order: {str(e)}")
            raise


