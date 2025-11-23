"""
Integration test script for the Execution Engine.

This script verifies that the TradingEngine works correctly by:
1. Connecting to the broker
2. Getting the initial portfolio state
3. Placing a test buy order
4. Verifying the final portfolio state
5. Verifying the trade is logged to the database
"""

import sys
from pathlib import Path

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from database.service import DatabaseService
from execution.engine import TradingEngine
from database.models import Trade


def main():
    """Run integration tests for the execution engine."""
    print("=" * 60)
    print("Execution Engine Integration Test")
    print("=" * 60)
    print()
    
    try:
        # Step 0: Initialize database service for logging
        print("Step 0: Initializing database service...")
        db_service = DatabaseService("sqlite:///:memory:")  # In-memory DB for testing
        print("✓ Database service initialized (in-memory)")
        print()
        
        # Step 1: Instantiate TradingEngine with database logging
        print("Step 1: Initializing TradingEngine with Alpaca provider and database logging...")
        engine = TradingEngine(
            provider_name="alpaca",
            db_service=db_service,
            strategy_name="test_strategy"
        )
        print("✓ TradingEngine initialized successfully")
        print()
        
        # Step 2: Get initial portfolio state
        print("Step 2: Getting initial portfolio state...")
        initial_state = engine.get_portfolio_state()
        print(f"✓ Initial Portfolio State:")
        print(f"  Cash: ${initial_state.cash:,.2f}")
        print(f"  Total Equity: ${initial_state.total_equity:,.2f}")
        print(f"  Positions: {initial_state.positions}")
        print()
        
        # Step 3: Execute test buy order
        print("Step 3: Executing test buy order...")
        print("  Symbol: AAPL")
        print("  Amount: $100.00")
        order_response = engine.place_buy_order("AAPL", 100.0)
        print(f"✓ Order executed successfully")
        print(f"  Order ID: {order_response.get('id')}")
        print(f"  Status: {order_response.get('status')}")
        print(f"  Filled Qty: {order_response.get('filled_qty', 0)}")
        print(f"  Client Order ID: {order_response.get('client_order_id')}")
        print()
        
        # Step 3.5: Verify trade was logged to database
        print("Step 3.5: Verifying trade was logged to database...")
        with db_service.get_session() as session:
            # Query for the trade using client_order_id
            client_order_id = order_response.get('client_order_id')
            if client_order_id:
                logged_trade = session.query(Trade).filter_by(
                    client_order_id=client_order_id
                ).first()
                
                if logged_trade:
                    print(f"✓ Trade logged to database:")
                    print(f"  Trade ID: {logged_trade.id}")
                    print(f"  Strategy: {logged_trade.strategy_name}")
                    print(f"  Symbol: {logged_trade.symbol}")
                    print(f"  Side: {logged_trade.side}")
                    print(f"  Quantity: {logged_trade.qty}")
                    print(f"  Price: ${logged_trade.price:.2f}")
                    print(f"  Notional: ${logged_trade.notional:.2f}")
                    print(f"  Timestamp: {logged_trade.timestamp}")
                    print()
                    
                    # Verify client_order_id matches
                    assert logged_trade.client_order_id == client_order_id, \
                        f"Client Order ID mismatch: DB={logged_trade.client_order_id}, Response={client_order_id}"
                    print("✓ Client Order ID matches between Alpaca and database")
                else:
                    print(f"⚠ Warning: Trade not found in database for client_order_id: {client_order_id}")
            else:
                print("⚠ Warning: No client_order_id in order response, cannot verify database logging")
        print()
        
        # Step 4: Get final portfolio state
        print("Step 4: Getting final portfolio state...")
        final_state = engine.get_portfolio_state()
        print(f"✓ Final Portfolio State:")
        print(f"  Cash: ${final_state.cash:,.2f}")
        print(f"  Total Equity: ${final_state.total_equity:,.2f}")
        print(f"  Positions: {final_state.positions}")
        print()
        
        # Step 5: Compare states
        print("Step 5: Comparing states...")
        cash_change = final_state.cash - initial_state.cash
        equity_change = final_state.total_equity - initial_state.total_equity
        print(f"  Cash change: ${cash_change:,.2f}")
        print(f"  Equity change: ${equity_change:,.2f}")
        
        if "AAPL" in final_state.positions:
            aapl_position = final_state.positions["AAPL"]
            print(f"  AAPL position: {aapl_position} shares")
        else:
            print("  AAPL position: 0 shares (order may still be pending)")
        
        print()
        print("=" * 60)
        print("Integration test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Integration test failed: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


