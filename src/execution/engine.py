"""
Trading Engine - Main service logic for order execution.

This module provides the TradingEngine class that orchestrates order execution
using the Factory Pattern to select the appropriate broker provider.
"""

import uuid
from typing import List, Optional

from .interface import ExecutionProvider
from .models import OrderSide, OrderType, PortfolioState, TargetPortfolio, TradeOrder
from .providers.alpaca import AlpacaProvider

# Minimum trade threshold to avoid "dust" trades
MIN_TRADE_THRESHOLD = 1.0  # dollars


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
            except Exception as e:
                print(f"✗ Error executing sell order for {sell_order.symbol}: {str(e)}")
                # Continue with other orders even if one fails
                continue
        
        # Execute all buys after sells complete
        print(f"Executing {len(buys)} buy order(s)...")
        for buy_order in buys:
            try:
                result = self.provider.execute_order(buy_order)
                executed_orders.append(result)
            except Exception as e:
                print(f"✗ Error executing buy order for {buy_order.symbol}: {str(e)}")
                # Continue with other orders even if one fails
                continue
        
        print(f"✓ Rebalance complete: {len(executed_orders)} order(s) executed")
        return executed_orders


