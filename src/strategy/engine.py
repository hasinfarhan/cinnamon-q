"""
Strategy Engine - The Orchestrator

This module provides the main StrategyEngine class that orchestrates the
complete strategy pipeline: Data → Signals → Allocation → Target Portfolio.

The StrategyEngine is the "brain in a box" - it doesn't know about brokers
or orders, only about data, signals, and target allocations. This makes it
perfectly testable in isolation.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from execution.models import TargetPortfolio
from .allocators.top_k import TopKEqualWeight
from .interfaces import Allocator, SignalGenerator
from .signals.momentum import MomentumSignal


class StrategyEngine:
    """
    Main strategy orchestrator that binds data, signal generation, and allocation.
    
    The StrategyEngine follows a pipeline architecture:
    Market Data → Signal Generator → Scores → Allocator → Target Portfolio
    
    This class is stateless and testable - it doesn't interact with brokers
    or execution systems, only with data and strategy logic.
    
    Attributes:
        signal_gen: Signal generator instance (e.g., MomentumSignal)
        allocator: Allocator instance (e.g., TopKEqualWeight)
        data_path: Path to the parquet file containing market data
    """
    
    def __init__(
        self,
        signal_gen: Optional[SignalGenerator] = None,
        allocator: Optional[Allocator] = None,
        data_path: Optional[str] = None
    ):
        """
        Initialize the StrategyEngine.
        
        Args:
            signal_gen: Signal generator instance. If None, defaults to MomentumSignal(window_days=30)
            allocator: Allocator instance. If None, defaults to TopKEqualWeight(k=3)
            data_path: Path to parquet file containing market data.
                Expected columns: ['date', 'ticker', 'close']
                If None, defaults to 'data/processed/market_data.parquet'
        """
        # Default signal generator: MomentumSignal with 30-day window
        self.signal_gen = signal_gen if signal_gen is not None else MomentumSignal(window_days=30)
        
        # Default allocator: TopKEqualWeight with k=3
        self.allocator = allocator if allocator is not None else TopKEqualWeight(k=3)
        
        # Default data path
        if data_path is None:
            # Try to find data relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_path = str(project_root / "data" / "processed" / "market_data.parquet")
        
        self.data_path = Path(data_path)
        
        # Validate data path exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please ensure the parquet file exists at the specified path."
            )
    
    def get_target_portfolio(self, current_time: Optional[pd.Timestamp] = None) -> TargetPortfolio:
        """
        Generate target portfolio based on current strategy and data.
        
        This method orchestrates the complete strategy pipeline:
        1. Load market data from parquet file
        2. Generate signals using the signal generator
        3. Allocate capital using the allocator
        4. Return validated TargetPortfolio
        
        Args:
            current_time: Optional timestamp to filter data (for future use).
                Currently not used - loads all available data.
                
        Returns:
            TargetPortfolio: Validated Pydantic model containing target weights.
                This can be passed directly to ExecutionEngine.rebalance()
                
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        # Step 1: Load data from parquet file
        print(f"📊 Brain: Loading data from {self.data_path}...")
        try:
            data = pd.read_parquet(self.data_path)
            print(f"   ✓ Loaded {len(data)} rows of data")
        except Exception as e:
            print(f"   ✗ Error loading data: {str(e)}")
            raise
        
        # Validate data has required columns
        required_columns = ['date', 'ticker', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
        
        # Step 2: Generate signals using signal generator
        print(f"🔍 Brain: Generating signals using {type(self.signal_gen).__name__}...")
        try:
            signals = self.signal_gen.generate(data)
            num_assets = len(signals)
            print(f"   ✓ Generated signals for {num_assets} assets")
        except Exception as e:
            print(f"   ✗ Error generating signals: {str(e)}")
            raise
        
        # Step 3: Allocate capital using allocator
        print(f"💰 Brain: Allocating capital using {type(self.allocator).__name__}...")
        try:
            target = self.allocator.allocate(signals)
            selected_assets = list(target.weights.keys())
            num_selected = len(selected_assets)
            
            if num_selected > 0:
                print(f"   ✓ Selected {num_selected} assets: {selected_assets}")
                print(f"   ✓ Target weights: {target.weights}")
            else:
                print(f"   ✓ No assets selected (Bear Market - 100% Cash/Safety)")
        except Exception as e:
            print(f"   ✗ Error allocating capital: {str(e)}")
            raise
        
        # Step 4: Return validated TargetPortfolio
        print(f"✅ Brain: Strategy pipeline complete -> Target Portfolio ready")
        return target

