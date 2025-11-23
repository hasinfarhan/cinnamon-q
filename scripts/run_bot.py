"""
Production Bot - End-to-End Integration

This script connects the Strategy Engine (Brain) to the Trading Engine (Execution),
creating a complete automated trading system.

Pipeline:
1. Data Refresh: Verify market data exists
2. The Brain: StrategyEngine generates TargetPortfolio
3. The Execution: TradingEngine rebalances to target

This is the prototype of the Production Bot.
"""

import sys
from pathlib import Path

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config.loader import load_config
from execution.engine import TradingEngine
from strategy.allocators.top_k import TopKEqualWeight
from strategy.engine import StrategyEngine
from strategy.signals.momentum import MomentumSignal


def main():
    """Run the complete trading bot pipeline."""
    print("=" * 80)
    print("🤖 Production Bot - End-to-End Integration")
    print("=" * 80)
    print()
    
    # Step 0: Load Configuration
    print("Step 0: Loading configuration...")
    try:
        config = load_config()
        print(f"✓ Configuration loaded from config.yaml")
        print(f"  Heartbeat interval: {config.system.heartbeat_interval}s")
        print(f"  Strategies: {len(config.strategies)}")
        print()
    except Exception as e:
        print(f"✗ Error loading configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 1: Data Refresh - Verify market data exists
    print("Step 1: Checking market data...")
    # Use config data path if available, otherwise default
    if config.data and config.data.output_path:
        data_path = project_root / config.data.output_path
    else:
        data_path = project_root / "data" / "processed" / "market_data.parquet"
    
    if not data_path.exists():
        print(f"✗ Error: Market data file not found: {data_path}")
        print()
        print("Please run the data ingestion script first:")
        print("  python scripts/01_fetch_market_data.py")
        print()
        print("Or ensure the parquet file exists at:")
        print(f"  {data_path}")
        sys.exit(1)
    
    print(f"✓ Market data found: {data_path}")
    print()
    
    # Step 2: The Brain - Strategy Engine
    print("Step 2: Initializing Strategy Engine (The Brain)...")
    print("-" * 80)
    
    try:
        # Get first strategy from config (for MVP, we use the first one)
        if not config.strategies:
            raise ValueError("No strategies defined in config.yaml")
        
        strategy_config = config.strategies[0]
        print(f"✓ Using strategy: {strategy_config.name}")
        
        # Initialize signal generator based on config
        if strategy_config.signal == "momentum":
            window = strategy_config.params.get("window", 30)
            signal_gen = MomentumSignal(window_days=window)
            print(f"✓ Signal Generator: {type(signal_gen).__name__} (window={window} days)")
        else:
            raise ValueError(f"Unknown signal type: {strategy_config.signal}")
        
        # Initialize allocator based on config
        k = strategy_config.params.get("k", 3)
        allocator = TopKEqualWeight(k=k)
        print(f"✓ Allocator: {type(allocator).__name__} (k={k})")
        
        # Initialize StrategyEngine
        brain = StrategyEngine(
            signal_gen=signal_gen,
            allocator=allocator,
            data_path=str(data_path)
        )
        print(f"✓ StrategyEngine initialized")
        print()
        
        # Get target portfolio from the Brain
        print("Running strategy pipeline...")
        print()
        target = brain.get_target_portfolio()
        print()
        
        # Display target portfolio
        print("=" * 80)
        print("🎯 Target Portfolio:")
        if target.weights:
            for ticker, weight in target.weights.items():
                print(f"  {ticker:10s}: {weight:6.2%}")
            print(f"  {'Total':10s}: {sum(target.weights.values()):6.2%}")
        else:
            print("  {} (100% Cash/Safety - Bear Market Detected)")
        print("=" * 80)
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Strategy Engine Error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: The Execution - Trading Engine
    print("Step 3: Initializing Trading Engine (The Execution)...")
    print("-" * 80)
    
    try:
        # Initialize TradingEngine with provider from config
        engine = TradingEngine(provider_name=config.execution.provider)
        print(f"✓ TradingEngine initialized with {config.execution.provider} provider")
        print(f"  Paper trading: {config.execution.paper}")
        print()
        
        # Get current portfolio state
        current_state = engine.get_portfolio_state()
        print("Current Portfolio State:")
        print(f"  Cash: ${current_state.cash:,.2f}")
        print(f"  Total Equity: ${current_state.total_equity:,.2f}")
        print(f"  Positions: {current_state.positions}")
        print()
        
        # Execute rebalance to target
        print("Executing rebalance to target portfolio...")
        print()
        orders = engine.rebalance(target)
        print()
        
        # Display execution results
        print("=" * 80)
        print("📊 Execution Results:")
        if orders:
            print(f"  Executed {len(orders)} order(s):")
            for i, order in enumerate(orders, 1):
                symbol = order.get('symbol', 'UNKNOWN')
                side = order.get('side', 'UNKNOWN')
                status = order.get('status', 'UNKNOWN')
                order_id = order.get('id', 'N/A')
                print(f"    {i}. {side.upper():4s} {symbol:10s} | Status: {status:10s} | ID: {order_id}")
        else:
            print("  No trades executed (already at target or below threshold)")
        print("=" * 80)
        print()
        
        # Get final portfolio state
        final_state = engine.get_portfolio_state()
        print("Final Portfolio State:")
        print(f"  Cash: ${final_state.cash:,.2f}")
        print(f"  Total Equity: ${final_state.total_equity:,.2f}")
        print(f"  Positions: {final_state.positions}")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Trading Engine Error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success summary
    print("=" * 80)
    print("✅ Bot execution completed successfully!")
    print("=" * 80)
    print()
    print("Pipeline Summary:")
    print("  1. ✓ Data loaded and validated")
    print("  2. ✓ Signals generated from market data")
    print("  3. ✓ Portfolio allocated with Halal constraints")
    print("  4. ✓ Trades executed to rebalance portfolio")
    print()
    print("The Brain and Body are now connected! 🧠💪")


if __name__ == "__main__":
    main()

