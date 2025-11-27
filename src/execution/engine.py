"""
Trading Engine - Main service logic for order execution.

This module provides the TradingEngine class that orchestrates order execution
using the Factory Pattern to select the appropriate broker provider.
"""

import time
import uuid
from typing import List, Optional

from .interface import ExecutionProvider
from .models import OrderSide, OrderType, PortfolioState, TargetPortfolio, TradeOrder
from .providers.alpaca import AlpacaProvider

# Optional import for database logging
try:
    from database.service import DatabaseService
except ImportError:
    DatabaseService = None  # type: ignore

# Minimum trade threshold to avoid "dust" trades
# Alpaca requires minimum $10 notional for crypto orders
MIN_TRADE_THRESHOLD = 10.0  # dollars (Alpaca minimum)


class TradingEngine:
    """
    Main trading engine that provides a high-level interface for order execution.
    
    Uses the Factory Pattern to instantiate the appropriate execution provider
    based on the provider name.
    """
    
    def __init__(
        self,
        provider_name: str = "alpaca",
        db_service: Optional["DatabaseService"] = None,
        strategy_name: str = "momentum_bluechip"
    ):
        """
        Initialize the TradingEngine with a specific provider.
        
        Args:
            provider_name: Name of the provider to use (default: "alpaca")
            db_service: Optional DatabaseService instance for logging trades.
                If None, trades will not be logged to the database.
            strategy_name: Name of the strategy generating trades (default: "momentum_bluechip")
                Used for trade logging to identify which strategy executed the trade.
            
        Raises:
            ValueError: If provider_name is not supported
        """
        self.provider = self._create_provider(provider_name)
        self.db = db_service
        self.strategy_name = strategy_name
        
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
        result = self.provider.execute_order(order)
        
        # Log trade to database if db_service is available
        if self.db is not None and result is not None:
            self._log_trade_to_db(order, result)
        
        return result
    
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
        result = self.provider.execute_order(order)
        
        # Log trade to database if db_service is available
        if self.db is not None and result is not None:
            self._log_trade_to_db(order, result)
        
        return result
    
    def rebalance(self, target: TargetPortfolio) -> List[dict]:
        """
        Rebalance the portfolio to match the target weights.
        
        This method compares the target portfolio weights against the current
        portfolio state and executes the necessary trades to achieve the target
        allocation. Trades are sequenced: all sells execute first, then all buys.
        
        Args:
            target: TargetPortfolio with desired weight distribution
            
        Returns:
            List[dict]: List of order execution responses from the broker
            
        Raises:
            ValueError: If target validation fails
            Exception: If order execution fails
        """
        # Step A: Fetch current portfolio state
        state = self.get_portfolio_state()
        
        # Step B: Identify all relevant tickers (union of target weights and current positions)
        all_tickers = set(target.weights.keys()) | set(state.positions.keys())
        
        # Step C: Calculate differences and build order lists
        sells: List[TradeOrder] = []
        buys: List[TradeOrder] = []
        
        for ticker in all_tickers:
            # Calculate target dollar value
            target_weight = target.weights.get(ticker, 0.0)
            target_equity = state.total_equity * target_weight
            
            # Get current dollar value and quantity
            current_equity = state.valuations.get(ticker, 0.0)
            current_qty = state.positions.get(ticker, 0.0)
            
            # Calculate difference
            diff = target_equity - current_equity
            
            # Step D: Filter out small "dust" trades
            if abs(diff) < MIN_TRADE_THRESHOLD:
                continue
            
            # Generate unique client_order_id for idempotency
            client_order_id = f"rebalance_{ticker}_{uuid.uuid4().hex[:8]}"
            
            # Step E: Create orders based on difference
            if diff < 0:
                # Need to sell: diff is negative
                # FIX: Handle Full Exit (Sell Everything) - Use QTY to avoid price slippage
                # When target_weight is 0.0, we want to close the position completely
                # Using qty ensures we sell exactly what we have, regardless of price movements
                if target_weight == 0.0:
                    print(f"📉 Closing position for {ticker} completely (target weight: 0.0)")
                    sell_order = TradeOrder(
                        symbol=ticker,
                        side=OrderSide.SELL,
                        qty=abs(current_qty),  # Sell exact quantity, not dollar amount
                        order_type=OrderType.MARKET,
                        client_order_id=client_order_id
                    )
                else:
                    # Partial Sell (Rebalance) - Use Notional
                    sell_order = TradeOrder(
                        symbol=ticker,
                        side=OrderSide.SELL,
                        notional=abs(diff),
                        order_type=OrderType.MARKET,
                        client_order_id=client_order_id
                    )
                sells.append(sell_order)
            elif diff > 0:
                # Need to buy: diff is positive
                # Buys are always Notional (Dollar based)
                buy_order = TradeOrder(
                    symbol=ticker,
                    side=OrderSide.BUY,
                    notional=diff,
                    order_type=OrderType.MARKET,
                    client_order_id=client_order_id
                )
                buys.append(buy_order)
        
        # Step F: Execute orders (sells first, then buys)
        executed_orders: List[dict] = []
        
        # Execute all sells first (crucial for cash management)
        print(f"Executing {len(sells)} sell order(s)...")
        for sell_order in sells:
            try:
                result = self.provider.execute_order(sell_order)
                executed_orders.append(result)
                
                # Log trade to database if db_service is available
                if self.db is not None and result is not None:
                    self._log_trade_to_db(sell_order, result)
                
                # Rate limiting: Prevent API rate limit hits when rebalancing many assets
                # Alpaca limit: 200 req/min = ~3.3 req/sec, so 0.1s delay is safe
                # This is especially important for v1.2 with 17+ assets
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"✗ Error executing sell order for {sell_order.symbol}: {str(e)}")
                # Continue with other orders even if one fails
                continue
        
        # Execute all buys after sells complete
        # First, check available cash and filter out orders that exceed balance
        state_after_sells = self.get_portfolio_state()  # Refresh state after sells
        available_cash = state_after_sells.cash
        
        # Filter buys: Remove orders below minimum threshold or exceeding available cash
        valid_buys: List[TradeOrder] = []
        total_buy_notional = 0.0
        
        for buy_order in buys:
            notional = buy_order.notional or 0.0
            
            # Skip if below minimum threshold
            if notional < MIN_TRADE_THRESHOLD:
                print(f"  ⚠ Skipping {buy_order.symbol}: ${notional:.2f} below ${MIN_TRADE_THRESHOLD} minimum")
                continue
            
            # Check if we have enough cash (with 5% buffer for fees/slippage)
            if total_buy_notional + notional > available_cash * 0.95:
                print(f"  ⚠ Skipping {buy_order.symbol}: Would exceed available cash (${available_cash:.2f})")
                continue
            
            valid_buys.append(buy_order)
            total_buy_notional += notional
        
        print(f"Executing {len(valid_buys)} buy order(s) (filtered from {len(buys)} total, ${total_buy_notional:.2f} total notional)...")
        for buy_order in valid_buys:
            try:
                result = self.provider.execute_order(buy_order)
                executed_orders.append(result)
                
                # Log trade to database if db_service is available
                if self.db is not None and result is not None:
                    self._log_trade_to_db(buy_order, result)
                
                # Rate limiting: Prevent API rate limit hits when rebalancing many assets
                # Alpaca limit: 200 req/min = ~3.3 req/sec, so 0.1s delay is safe
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"✗ Error executing buy order for {buy_order.symbol}: {str(e)}")
                # Continue with other orders even if one fails
                continue
        
        print(f"✓ Rebalance complete: {len(executed_orders)} order(s) executed")
        return executed_orders
    
    def _log_trade_to_db(self, order: TradeOrder, result: dict) -> None:
        """
        Log a trade execution to the database.
        
        This is an internal helper method that extracts trade information
        from the order and broker response, then logs it to the database.
        
        For MVP, we log the order submission (not the fill) since fills
        happen asynchronously. This provides an audit trail of "What did
        the bot decide to do?" which is the core of the Mudarabah audit trail.
        
        Args:
            order: The TradeOrder that was executed
            result: The raw response dict from the broker API
        """
        if self.db is None:
            return
        
        try:
            # Extract trade details from order and response
            symbol = order.symbol
            side = order.side.value  # "buy" or "sell"
            
            # Get quantity: prefer filled_qty if available, otherwise use requested qty
            # Note: For pending orders, filled_qty will be 0, so we use qty from order response
            qty = result.get("filled_qty") or result.get("qty") or order.qty or 0.0
            # If order is pending, use the requested qty from the order itself
            if qty == 0.0 and order.qty is not None:
                qty = order.qty
            elif qty == 0.0 and order.notional is not None:
                # Estimate qty from notional (will be updated when order fills)
                # For logging purposes, we'll use 0.0 and note it's pending
                qty = 0.0
            if qty is None:
                qty = 0.0
            # Ensure qty is a number (convert string to float if needed)
            try:
                qty = float(qty) if qty != 0.0 else 0.0
            except (ValueError, TypeError):
                qty = float(order.qty) if order.qty is not None else 0.0
            
            # Get price: prefer filled_avg_price if available, otherwise use limit_price or 0.0
            # For market orders, we might not have a price yet (order is pending)
            # For MVP, we log the order intent, so use limit_price if available, otherwise 0.0
            price = (
                result.get("filled_avg_price") or
                result.get("limit_price") or
                order.limit_price or
                0.0
            )
            if price is None:
                price = 0.0
            # Ensure price is a number (convert string to float if needed)
            try:
                price = float(price) if price != 0.0 else 0.0
            except (ValueError, TypeError):
                price = 0.0
            
            # Calculate notional: qty * price, or use order.notional if available
            if order.notional is not None and order.notional > 0:
                notional = float(order.notional)
            elif qty > 0 and price > 0:
                notional = qty * price
            else:
                notional = 0.0
            
            # Get client_order_id: must be present for idempotency
            client_order_id = result.get("client_order_id") or order.client_order_id
            if not client_order_id:
                # Fallback: generate one (shouldn't happen, but safety check)
                client_order_id = f"log_{symbol}_{uuid.uuid4().hex[:8]}"
            
            # Log to database
            self.db.log_trade(
                strategy_name=self.strategy_name,
                symbol=symbol,
                side=side,
                qty=float(qty),
                price=float(price),
                notional=float(notional),
                client_order_id=client_order_id
            )
            
            print(f"  📝 Trade logged to database: {side.upper()} {qty} {symbol} @ ${price:.2f}")
            
        except Exception as e:
            # Don't fail the trade execution if logging fails
            print(f"  ⚠ Warning: Failed to log trade to database: {str(e)}")


