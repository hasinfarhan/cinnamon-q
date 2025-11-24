"""
Alpaca Trading API Provider Implementation.

This module implements the ExecutionProvider interface using the Alpaca API.
"""

# Import compatibility fixes BEFORE alpaca-trade-api
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
try:
    from compat import _patch_urllib3
    _patch_urllib3()
except ImportError:
    pass

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
            
            # ROUNDING FIX v2:
            # 1. Notional (Dollars): Round to 2 decimals (Alpaca requirement)
            #    Alpaca rejects high precision floats for dollar amounts
            # 2. Quantity (Shares/Coins): Do NOT round
            #    Alpaca accepts high precision (up to 9 decimals for crypto)
            #    Rounding qty can cause "insufficient balance" errors when selling
            #    Example: 20.776191583 ETH rounded to 20.78 tries to sell more than owned
            
            # Handle notional orders: alpaca-trade-api doesn't support notional directly,
            # so we need to fetch the current price and calculate quantity
            qty = order.qty
            notional = round(order.notional, 2) if order.notional is not None else None
            
            # If notional is provided, we need to calculate qty from current price
            if notional is not None and qty is None:
                try:
                    # Try to get last quote first (more accurate for current price)
                    try:
                        quote = self.api.get_last_quote(symbol)
                        if quote:
                            # Use bid price for sells, ask price for buys
                            if order.side == OrderSide.SELL:
                                current_price = float(quote.bp)  # bid price
                            else:
                                current_price = float(quote.ap)  # ask price
                            
                            # Calculate quantity from notional
                            qty = notional / current_price
                            print(f"  Calculated qty from quote: ${notional:.2f} / ${current_price:.2f} = {qty:.9f}")
                        else:
                            raise AttributeError("Quote is None")
                    except (AttributeError, KeyError):
                        # Fallback: try to get last trade price
                        last_trade = self.api.get_last_trade(symbol)
                        if last_trade:
                            current_price = float(last_trade.price)
                            qty = notional / current_price
                            print(f"  Calculated qty from last trade: ${notional:.2f} / ${current_price:.2f} = {qty:.9f}")
                        else:
                            raise ValueError(f"Could not fetch price for {symbol} to calculate quantity from notional")
                except Exception as e:
                    print(f"⚠ Warning: Could not calculate qty from notional: {e}")
                    raise ValueError(f"Failed to calculate quantity from notional ${notional:.2f} for {symbol}: {e}")
            
            # Build order parameters (alpaca-trade-api requires positional args, not kwargs)
            # submit_order(symbol, qty, side, type, time_in_force, limit_price=None, ...)
            submit_kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": order.side.value,
                "type": order.order_type.value,
                "time_in_force": time_in_force,
            }
            
            # Add limit price if it's a limit order
            if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                submit_kwargs["limit_price"] = round(order.limit_price, 2)
            
            # CRITICAL: Add client_order_id for idempotency
            if order.client_order_id:
                submit_kwargs["client_order_id"] = order.client_order_id
            
            # Submit the order
            print(f"  Symbol: {symbol} | Side: {order.side.value} | Type: {order.order_type.value} | TIF: {time_in_force}")
            if qty is not None:
                print(f"  Quantity: {qty:.9f}")
            if notional is not None:
                print(f"  Notional: ${notional:.2f}")
            if order.limit_price is not None:
                print(f"  Limit Price: ${order.limit_price:.2f}")
            if order.client_order_id:
                print(f"  Client Order ID: {order.client_order_id}")
            
            response = self.api.submit_order(**submit_kwargs)
            
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


