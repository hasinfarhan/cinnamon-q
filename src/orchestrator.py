"""
Bot Orchestrator - Main Application Lifecycle Manager

This module provides the BotOrchestrator class that manages the application
lifecycle, enforces strategy schedules (multi-timeframe), and ties together
the Config, Database, Strategy Engine, and Execution Engine.

The orchestrator runs a "Heartbeat Loop" that checks strategy schedules
and executes them when their intervals have elapsed.
"""

import logging
import multiprocessing
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config.loader import load_config
from config.schema import StrategyConfig
from database.service import DatabaseService
from execution.engine import TradingEngine
from pipeline.runner import run_full_pipeline
from strategy.allocators.top_k import TopKEqualWeight
from strategy.engine import StrategyEngine
from strategy.ppo_strategy import PPOStrategy
from strategy.signals.momentum import MomentumSignal

logger = logging.getLogger(__name__)


class BotOrchestrator:
    """
    Main orchestrator that manages the trading bot lifecycle.
    
    This class:
    - Loads configuration from YAML
    - Initializes all services (Database, Trading Engine, Strategy Engines)
    - Manages strategy schedules (multi-timeframe support)
    - Executes strategies when their intervals have elapsed
    - Runs the main heartbeat loop
    
    Attributes:
        config: Loaded AppConfig instance
        db: DatabaseService instance
        trading_engine: TradingEngine instance
        strategies: Dictionary mapping strategy names to StrategyEngine instances
        last_run: Dictionary mapping strategy names to last execution timestamps
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the BotOrchestrator.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default search paths.
        """
        print("=" * 80)
        print("🤖 Initializing arynosQ Orchestrator...")
        print("=" * 80)
        print()
        
        # Step 1: Load configuration
        print("Step 1: Loading configuration...")
        self.config = load_config(config_path)
        print(f"✓ Configuration loaded")
        print(f"  Heartbeat interval: {self.config.system.heartbeat_interval}s")
        print(f"  Strategies: {len(self.config.strategies)}")
        print(f"  Execution provider: {self.config.execution.provider}")
        print()
        
        # Step 2: Initialize Database Service
        print("Step 2: Initializing database service...")
        self.db = DatabaseService(
            connection_url=self.config.system.db_url,
            echo=False  # Set to True for SQL debugging
        )
        print(f"✓ Database service initialized: {self.config.system.db_url}")
        print()
        
        # Step 3: Initialize Trading Engine
        print("Step 3: Initializing trading engine...")
        self.trading_engine = TradingEngine(
            provider_name=self.config.execution.provider,
            db_service=self.db,
            strategy_name="momentum_bluechip"  # Default, will be overridden per strategy
        )
        print(f"✓ Trading engine initialized: {self.config.execution.provider}")
        print()
        
        # Step 4: Initialize Strategy Engines
        print("Step 4: Initializing strategy engines...")
        self.strategies: Dict[str, any] = {}  # Can be StrategyEngine or PPOStrategy
        for strategy_config in self.config.strategies:
            strategy_engine = self._create_strategy_engine(strategy_config)
            self.strategies[strategy_config.name] = strategy_engine
            print(f"✓ Strategy '{strategy_config.name}' initialized")
            print(f"    Signal: {strategy_config.signal}, Interval: {strategy_config.interval}")
            print(f"    Params: {strategy_config.params}")
        print()
        
        # Step 5: Initialize last run times (empty = never run, will run immediately)
        self.last_run: Dict[str, datetime] = {}
        
        # Step 6: Initialize Training Scheduler (Background MLOps Pipeline)
        self.training_interval_hours = self.config.system.training_interval_hours
        self.last_training_run: Optional[datetime] = None
        self.training_process: Optional[multiprocessing.Process] = None
        
        if self.training_interval_hours > 0:
            print(f"✓ Training scheduler initialized: Every {self.training_interval_hours} hours")
        else:
            print("✓ Training scheduler disabled (training_interval_hours = 0)")
        
        print("=" * 80)
        print("✅ Orchestrator initialization complete!")
        print("=" * 80)
        print()
    
    def _create_strategy_engine(self, strategy_config: StrategyConfig):
        """
        Create a StrategyEngine or PPOStrategy instance from a StrategyConfig.
        
        This method unpacks the strategy configuration and creates the
        appropriate strategy instance based on the signal type.
        
        Args:
            strategy_config: StrategyConfig from the YAML file
        
        Returns:
            StrategyEngine or PPOStrategy: Configured strategy instance
        """
        # Handle PPO strategy
        if strategy_config.signal == "ppo":
            # Extract PPO-specific parameters
            params = strategy_config.params
            model_path = Path(params.get("model_path", "models/hdrl_optimized/final_model.zip"))
            vec_norm_path = Path(params.get("vec_norm_path", "models/hdrl_optimized/vec_normalize.pkl"))
            data_path = Path(params.get("data_path", "data/processed/crypto_training_data.parquet"))
            
            # Convert relative paths to absolute
            if not model_path.is_absolute():
                model_path = project_root / model_path
            if not vec_norm_path.is_absolute():
                vec_norm_path = project_root / vec_norm_path
            if not data_path.is_absolute():
                data_path = project_root / data_path
            
            # Extract optimizer config (for Bicameral Brain v2.1 - Sortino Edition)
            optimizer_config = None
            if self.config.optimizer:
                optimizer_config = {
                    "lookback_days": self.config.optimizer.lookback_days,
                    "benchmark_rate": self.config.optimizer.benchmark_rate,
                    "risk_free_rate": self.config.optimizer.risk_free_rate
                }
            
            # Create and return PPOStrategy
            # Inject the trading engine's provider for live data fetching
            # This enables the "Live Data Bridge" - the strategy will fetch
            # real-time market data instead of using static parquet files
            return PPOStrategy(
                model_path=model_path,
                vec_norm_path=vec_norm_path,
                data_path=data_path,
                provider=self.trading_engine.provider,  # Enable live data fetching
                optimizer_config=optimizer_config  # Bicameral Brain optimizer config
            )
        
        # Handle traditional StrategyEngine (momentum, etc.)
        # Determine data path
        if self.config.data and self.config.data.output_path:
            data_path = self.config.data.output_path
        else:
            data_path = "data/processed/market_data.parquet"
        
        # Create signal generator based on strategy type
        signal_gen = None
        if strategy_config.signal == "momentum":
            # Extract window parameter (default: 30 days)
            window = strategy_config.params.get("window", 30)
            signal_gen = MomentumSignal(window_days=window)
        else:
            raise ValueError(f"Unsupported signal type: {strategy_config.signal}")
        
        # Create allocator
        allocator = None
        # For MVP, we only support TopKEqualWeight
        # Extract k parameter (default: 3)
        k = strategy_config.params.get("k", 3)
        allocator = TopKEqualWeight(k=k)
        
        # Create and return StrategyEngine
        return StrategyEngine(
            signal_gen=signal_gen,
            allocator=allocator,
            data_path=data_path
        )
    
    def _parse_interval(self, interval_str: str) -> timedelta:
        """
        Parse an interval string into a timedelta.
        
        Supports formats:
        - "1m", "5m", "15m", "30m" (minutes)
        - "1h", "4h" (hours)
        - "1d", "7d" (days)
        - "1w" (weeks)
        
        Args:
            interval_str: Interval string (e.g., "1d", "1h", "30m")
        
        Returns:
            timedelta: Corresponding timedelta object
        
        Raises:
            ValueError: If interval string format is invalid
        """
        interval_str = interval_str.lower().strip()
        
        # Extract number and unit
        if len(interval_str) < 2:
            raise ValueError(f"Invalid interval format: {interval_str}")
        
        # Get the last character (unit) and the rest (number)
        unit = interval_str[-1]
        try:
            number = int(interval_str[:-1])
        except ValueError:
            raise ValueError(f"Invalid interval number: {interval_str[:-1]}")
        
        # Map unit to timedelta
        if unit == 'm':
            return timedelta(minutes=number)
        elif unit == 'h':
            return timedelta(hours=number)
        elif unit == 'd':
            return timedelta(days=number)
        elif unit == 'w':
            return timedelta(weeks=number)
        else:
            raise ValueError(f"Unsupported interval unit: {unit}. Supported: m, h, d, w")
    
    def should_run(self, strategy_name: str, interval_str: str) -> bool:
        """
        Check if a strategy should run based on its interval.
        
        A strategy should run if:
        - It has never run before (not in last_run dict)
        - The time elapsed since last run >= interval
        
        Args:
            strategy_name: Name of the strategy
            interval_str: Interval string (e.g., "1d", "1h")
        
        Returns:
            bool: True if strategy should run, False otherwise
        """
        # If never run, should run immediately
        if strategy_name not in self.last_run:
            return True
        
        # Parse interval
        interval = self._parse_interval(interval_str)
        
        # Check if enough time has elapsed
        last_run_time = self.last_run[strategy_name]
        now = datetime.now(timezone.utc)
        time_elapsed = now - last_run_time
        
        return time_elapsed >= interval
    
    def run_strategy(self, strategy_config: StrategyConfig) -> None:
        """
        Execute a strategy: Data -> Signals -> Allocation -> Execution.
        
        This is the core pipeline that:
        1. Gets target portfolio from StrategyEngine
        2. Rebalances portfolio using TradingEngine
        3. Updates last run time
        
        Args:
            strategy_config: StrategyConfig for the strategy to run
        """
        strategy_name = strategy_config.name
        print("=" * 80)
        print(f"🎯 Running strategy: {strategy_name}")
        print("=" * 80)
        print()
        
        try:
            # Get strategy engine
            strategy_engine = self.strategies[strategy_name]
            
            # Step 1: Get target portfolio from StrategyEngine
            print(f"Step 1: Generating target portfolio...")
            target = strategy_engine.get_target_portfolio()
            print()
            
            # Display target portfolio
            print("Target Portfolio:")
            if target.weights:
                total_weight = sum(target.weights.values())
                cash_weight = 1.0 - total_weight
                
                # Display all asset weights
                for ticker, weight in sorted(target.weights.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {ticker:10s}: {weight:6.2%}")
                
                # Explicitly show cash if there's any
                if cash_weight > 1e-6:  # Only show if > 0.0001%
                    print(f"  {'CASH':10s}: {cash_weight:6.2%}")
                
                # Show total (should always be 100%)
                print(f"  {'Total':10s}: {(total_weight + cash_weight):6.2%}")
            else:
                print("  {} (100% Cash/Safety - Bear Market Detected)")
            print()
            
            # Step 2: Rebalance using TradingEngine
            print(f"Step 2: Executing rebalance...")
            # Update trading engine strategy name for this execution
            self.trading_engine.strategy_name = strategy_name
            orders = self.trading_engine.rebalance(target)
            print()
            
            # Display execution results
            print("Execution Results:")
            if orders:
                print(f"  Executed {len(orders)} order(s)")
                for i, order in enumerate(orders, 1):
                    symbol = order.get('symbol', 'UNKNOWN')
                    side = order.get('side', 'UNKNOWN')
                    status = order.get('status', 'UNKNOWN')
                    print(f"    {i}. {side.upper():4s} {symbol:10s} | Status: {status}")
            else:
                print("  No trades executed (already at target or below threshold)")
            print()
            
            # Step 3: Update last run time
            self.last_run[strategy_name] = datetime.now(timezone.utc)
            
            print("=" * 80)
            print(f"✅ Strategy '{strategy_name}' execution complete!")
            print("=" * 80)
            print()
            
        except Exception as e:
            print()
            print("=" * 80)
            print(f"✗ Error executing strategy '{strategy_name}': {str(e)}")
            print("=" * 80)
            import traceback
            traceback.print_exc()
            print()
            # Don't update last_run on error - will retry on next heartbeat
    
    def _check_training_schedule(self):
        """
        Check if it's time to trigger background training.
        
        This method checks if the training interval has elapsed and
        spawns a background process to run the MLOps pipeline if needed.
        """
        if self.training_interval_hours <= 0:
            return  # Training disabled
        
        now = datetime.now(timezone.utc)
        
        # Check if training is needed
        if self.last_training_run is None:
            # First run, trigger immediately
            should_train = True
        else:
            # Check if interval has elapsed
            elapsed = now - self.last_training_run
            should_train = elapsed >= timedelta(hours=self.training_interval_hours)
        
        # Check if previous training process is still running
        if self.training_process is not None and self.training_process.is_alive():
            logger.debug("Training process still running, skipping new training trigger")
            return
        
        if should_train:
            logger.info("=" * 80)
            logger.info("🤖 Triggering background MLOps pipeline...")
            logger.info(f"  Last training: {self.last_training_run or 'Never'}")
            logger.info(f"  Interval: {self.training_interval_hours} hours")
            logger.info("=" * 80)
            
            # Spawn background process for training
            # This allows the trading loop to continue uninterrupted
            self.training_process = multiprocessing.Process(
                target=_run_training_pipeline,
                name="MLOpsPipeline"
            )
            self.training_process.start()
            
            # Update last training run time
            self.last_training_run = now
            
            logger.info(f"✓ Training pipeline started (PID: {self.training_process.pid})")
            logger.info("  Trading will continue uninterrupted in the foreground")
    
    def start(self) -> None:
        """
        Start the main heartbeat loop.
        
        This method runs indefinitely, checking strategy schedules and
        executing them when their intervals have elapsed. It also manages
        background MLOps pipeline training on a schedule.
        
        The loop:
        1. Iterates through all configured strategies
        2. Checks if each strategy should run (based on interval)
        3. Executes strategies that are due
        4. Checks training schedule (spawns background process if needed)
        5. Sleeps for heartbeat_interval seconds
        6. Repeats
        
        Handles KeyboardInterrupt (Ctrl+C) gracefully for clean shutdown.
        """
        print("=" * 80)
        print("🚀 Starting arynosQ Orchestrator...")
        print("=" * 80)
        print()
        print("Features:")
        print(f"  - Strategy execution: Every {self.config.system.heartbeat_interval}s")
        if self.training_interval_hours > 0:
            print(f"  - Background training: Every {self.training_interval_hours} hours")
        print()
        print("The bot is now running. Press Ctrl+C to stop.")
        print()
        
        try:
            while True:
                # Iterate through all strategies
                for strategy_config in self.config.strategies:
                    strategy_name = strategy_config.name
                    interval = strategy_config.interval
                    
                    # Check if strategy should run
                    if self.should_run(strategy_name, interval):
                        self.run_strategy(strategy_config)
                    else:
                        # Calculate time until next run
                        if strategy_name in self.last_run:
                            last_run = self.last_run[strategy_name]
                            interval_delta = self._parse_interval(interval)
                            next_run = last_run + interval_delta
                            now = datetime.now(timezone.utc)
                            time_until_next = next_run - now
                            
                            if time_until_next.total_seconds() > 0:
                                hours = int(time_until_next.total_seconds() // 3600)
                                minutes = int((time_until_next.total_seconds() % 3600) // 60)
                                print(f"⏳ Strategy '{strategy_name}': Next run in {hours}h {minutes}m")
                
                # Check training schedule (non-blocking)
                if self.training_interval_hours > 0:
                    self._check_training_schedule()
                
                # Sleep for heartbeat interval
                print(f"💓 Heartbeat... waiting {self.config.system.heartbeat_interval}s")
                print()
                time.sleep(self.config.system.heartbeat_interval)
                
        except KeyboardInterrupt:
            print()
            print("=" * 80)
            print("🛑 Shutting down gracefully...")
            print("=" * 80)
            print()
            
            # Wait for training process to finish (with timeout)
            if self.training_process is not None and self.training_process.is_alive():
                print("Waiting for background training to complete (max 60s)...")
                self.training_process.join(timeout=60)
                if self.training_process.is_alive():
                    print("⚠ Training process still running, terminating...")
                    self.training_process.terminate()
                    self.training_process.join()
            
            print("✓ Orchestrator stopped. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print()
            print("=" * 80)
            print(f"✗ Fatal error in orchestrator: {str(e)}")
            print("=" * 80)
            import traceback
            traceback.print_exc()
            
            # Cleanup training process
            if self.training_process is not None and self.training_process.is_alive():
                self.training_process.terminate()
                self.training_process.join()
            
            sys.exit(1)


def _run_training_pipeline():
    """
    Wrapper function for running the training pipeline in a separate process.
    
    This function is called by multiprocessing.Process and runs the full
    MLOps pipeline in the background without blocking the main trading loop.
    """
    # Configure logging for background process
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [MLOps] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "mlops_pipeline.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Run the full pipeline
        success = run_full_pipeline()
        if success:
            logging.info("✓ MLOps pipeline completed successfully")
        else:
            logging.warning("⚠ MLOps pipeline completed with warnings")
    except Exception as e:
        logging.error(f"✗ MLOps pipeline failed: {e}", exc_info=True)

