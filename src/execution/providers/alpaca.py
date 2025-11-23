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
            
            # 1. Get raw data
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            # 2. Parse Portfolio Value
            total_equity = float(account.portfolio_value)
            cash = float(account.cash)
            
            # 3. Parse Positions & Valuations
            position_dict: Dict[str, float] = {}
            valuations_dict: Dict[str, float] = {}
            
            for position in positions:
                # NORMALIZATION FIX: Alpaca returns "BTCUSD", but we want "BTC/USD"
                # If it ends in USD and isn't a stock (simple heuristic), add the slash
                symbol = position.symbol
                if symbol.endswith('USD') and '/' not in symbol and len(symbol) > 3:
                    symbol = f"{symbol[:-3]}/{symbol[-3:]}"
                
                qty = float(position.qty)
                market_value = float(position.market_value)
                
                position_dict[symbol] = qty
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
    
    def cancel_all_orders(self) -> None:
        """
        Cancel all open orders to free up buying power.
        
        This is useful for cleaning up "zombie orders" that may be holding
        cash reserves even when the market is closed or orders are pending.
        """
        try:
            if not self.api:
                raise RuntimeError("Not connected to Alpaca. Call connect() first.")
            
            print("🧹 Cleaning up open orders...")
            self.api.cancel_all_orders()
            
            # Sleep to let Alpaca process the cancellations
            import time
            time.sleep(2)
            
            print("✓ All open orders cancelled")
        except Exception as e:
            print(f"✗ Error cancelling orders: {str(e)}")
            # Don't raise - allow test to continue even if cancellation fails
    
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
            
            print(f"🚀 Service: Executing {order.side.value.upper()} {order.symbol}...")
            
            # SYMBOL NORMALIZATION: Convert "BTC/USD" to "BTCUSD" for Alpaca API
            # Alpaca uses "BTCUSD" format internally, but we normalize to "BTC/USD" in get_state()
            symbol = order.symbol
            if '/' in symbol:
                # Convert "BTC/USD" -> "BTCUSD" for API call
                symbol = symbol.replace('/', '')
            
            # TIF FIX: Handle both "BTC/USD" and "BTCUSD" formats
            # Crypto symbols either contain '/' or end with 'USD'
            is_crypto = '/' in order.symbol or order.symbol.endswith('USD')
            time_in_force = 'gtc' if is_crypto else 'day'
            
            # ROUNDING FIX: Round notional/qty to 2 decimals
            # Alpaca rejects high precision floats (e.g., $58683.257634 -> $58683.25)
            qty = round(order.qty, 2) if order.qty is not None else None
            notional = round(order.notional, 2) if order.notional is not None else None
            
            # Build order parameters
            order_params = {
                "symbol": symbol,
                "side": order.side.value,
                "type": order.order_type.value,
                "time_in_force": time_in_force,
            }
            
            # Add quantity or notional
            if qty is not None:
                order_params["qty"] = qty
            elif notional is not None:
                order_params["notional"] = notional
            
            # Add limit price if it's a limit order
            if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                order_params["limit_price"] = round(order.limit_price, 2)
            
            # CRITICAL: Add client_order_id for idempotency
            if order.client_order_id:
                order_params["client_order_id"] = order.client_order_id
            
            # Submit the order
            print(f"  Symbol: {symbol} | Side: {order.side.value} | Type: {order.order_type.value} | TIF: {time_in_force}")
            if qty is not None:
                print(f"  Quantity: {qty}")
            if notional is not None:
                print(f"  Notional: ${notional:.2f}")
            if order.limit_price is not None:
                print(f"  Limit Price: ${order.limit_price:.2f}")
            if order.client_order_id:
                print(f"  Client Order ID: {order.client_order_id}")
            
            response = self.api.submit_order(**order_params)
            
            # Convert response to dict (use _raw if available, otherwise build manually)
            if hasattr(response, '_raw'):
                result = response._raw
            else:
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
            print(f"❌ Order validation error: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Error executing order: {str(e)}")
            raise


