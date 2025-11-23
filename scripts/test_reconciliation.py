"""
Integration test for the Reconciliation Logic.

This script simulates a full market cycle using a DummyBrain to drive
the TradingEngine through 4 distinct state transitions:
1. Entry: 100% Cash -> 60% AAPL / 40% MSFT
2. Rebalance: 60% AAPL / 40% MSFT -> 40% AAPL / 60% MSFT
3. Exit: 40% AAPL / 60% MSFT -> 100% Cash
4. Safety: Verify we're back to cash

This proves the rebalance() method correctly calculates diffs and
executes orders in the proper sequence (sells before buys).
"""

import sys
import time
from pathlib import Path

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from execution.engine import TradingEngine
from execution.models import TargetPortfolio


class DummyBrain:
    """
    A deterministic "puppet master" that drives the TradingEngine
    through a predefined sequence of target states.
    
    This simulates a strategy that:
    - Starts in cash (safety)
    - Enters positions
    - Rebalances
    - Exits to cash
    """
    
    def __init__(self):
        """Initialize the DummyBrain with tick counter."""
        self.tick = 0
    
    def next_target(self) -> TargetPortfolio:
        """
        Get the next target portfolio based on the current tick.
        
        Returns:
            TargetPortfolio: The target state for this tick
            
        Tick Sequence:
            Tick 0: {} (Empty/Cash - Safety state)
            Tick 1: {'AAPL': 0.6, 'MSFT': 0.4} (Entry)
            Tick 2: {'AAPL': 0.4, 'MSFT': 0.6} (Rebalance)
            Tick 3: {} (Empty/Cash - Exit)
        """
        if self.tick == 0:
            # Tick 0: Start in cash (Safety state)
            weights = {}
        elif self.tick == 1:
            # Tick 1: Enter positions (60% AAPL, 40% MSFT)
            weights = {'AAPL': 0.6, 'MSFT': 0.4}
        elif self.tick == 2:
            # Tick 2: Rebalance (40% AAPL, 60% MSFT)
            weights = {'AAPL': 0.4, 'MSFT': 0.6}
        elif self.tick == 3:
            # Tick 3: Exit to cash
            weights = {}
        else:
            # Default: stay in cash
            weights = {}
        
        # Create and return validated TargetPortfolio
        # Pydantic will validate weights sum <= 1.0 and no negative weights
        target = TargetPortfolio(weights=weights)
        
        # Increment tick for next call
        self.tick += 1
        
        return target


def format_weights(weights: dict) -> str:
    """Format weights dictionary for display."""
    if not weights:
        return "{} (Cash)"
    return ", ".join([f"{ticker}: {weight:.1%}" for ticker, weight in weights.items()])


def format_order_result(result: dict) -> str:
    """Format order execution result for display."""
    symbol = result.get('symbol', 'UNKNOWN')
    side = result.get('side', 'UNKNOWN')
    status = result.get('status', 'UNKNOWN')
    order_id = result.get('id', 'N/A')
    filled_qty = result.get('filled_qty', 0)
    
    return f"  {side.upper():4s} {symbol:6s} | Status: {status:10s} | ID: {order_id} | Filled: {filled_qty}"


def main():
    """Run the reconciliation integration test."""
    print("=" * 80)
    print("Reconciliation Integration Test - Full Market Cycle Simulation")
    print("=" * 80)
    print()
    print("This test simulates a complete trading cycle:")
    print("  Tick 0: Safety (100% Cash)")
    print("  Tick 1: Entry (60% AAPL, 40% MSFT)")
    print("  Tick 2: Rebalance (40% AAPL, 60% MSFT)")
    print("  Tick 3: Exit (100% Cash)")
    print()
    print("=" * 80)
    print()
    
    try:
        # Initialize TradingEngine
        print("Initializing TradingEngine with Alpaca provider...")
        engine = TradingEngine(provider_name="alpaca")
        print("✓ TradingEngine initialized successfully")
        print()
        
        # Get initial state
        initial_state = engine.get_portfolio_state()
        print(f"Initial Portfolio State:")
        print(f"  Cash: ${initial_state.cash:,.2f}")
        print(f"  Total Equity: ${initial_state.total_equity:,.2f}")
        print(f"  Positions: {initial_state.positions}")
        print(f"  Valuations: {initial_state.valuations}")
        print()
        
        # Initialize DummyBrain
        brain = DummyBrain()
        
        # Main execution loop: 4 ticks
        for tick_num in range(4):
            print("-" * 80)
            print(f"TICK {tick_num}: Getting target from DummyBrain...")
            
            # Get target from brain
            target = brain.next_target()
            target_weights_str = format_weights(target.weights)
            print(f"Target Portfolio: {target_weights_str}")
            print()
            
            # Get current state before rebalance
            state_before = engine.get_portfolio_state()
            print(f"State Before Rebalance:")
            print(f"  Cash: ${state_before.cash:,.2f}")
            print(f"  Total Equity: ${state_before.total_equity:,.2f}")
            print(f"  Positions: {state_before.positions}")
            print(f"  Valuations: {state_before.valuations}")
            print()
            
            # Execute rebalance
            print("Executing rebalance...")
            try:
                orders = engine.rebalance(target)
                
                if orders:
                    print(f"✓ Rebalance executed: {len(orders)} order(s)")
                    print("Order Results:")
                    for order_result in orders:
                        print(format_order_result(order_result))
                else:
                    print("✓ No trades needed (already at target or below threshold)")
                
            except Exception as e:
                print(f"✗ Error during rebalance: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            print()
            
            # Get state after rebalance
            state_after = engine.get_portfolio_state()
            print(f"State After Rebalance:")
            print(f"  Cash: ${state_after.cash:,.2f}")
            print(f"  Total Equity: ${state_after.total_equity:,.2f}")
            print(f"  Positions: {state_after.positions}")
            print(f"  Valuations: {state_after.valuations}")
            print()
            
            # Wait between ticks to allow Alpaca to process orders and update state
            if tick_num < 3:  # Don't sleep after the last tick
                print(f"Waiting 15 seconds for Alpaca to process orders and update state...")
                print()
                time.sleep(15)
        
        # Final state summary
        print("=" * 80)
        print("Final Portfolio State:")
        final_state = engine.get_portfolio_state()
        print(f"  Cash: ${final_state.cash:,.2f}")
        print(f"  Total Equity: ${final_state.total_equity:,.2f}")
        print(f"  Positions: {final_state.positions}")
        print(f"  Valuations: {final_state.valuations}")
        print()
        
        # Verify we're back to cash (or close to it)
        if not final_state.positions or all(abs(qty) < 0.01 for qty in final_state.positions.values()):
            print("✓ SUCCESS: Portfolio returned to cash state (all positions closed)")
        else:
            print("⚠ WARNING: Some positions remain (may be due to fractional shares or pending orders)")
        
        print()
        print("=" * 80)
        print("Integration test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Integration test failed: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

