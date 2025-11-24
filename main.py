"""
Main Entry Point - arynosQ Trading Bot

This is the main entry point for the trading bot. It initializes the
BotOrchestrator and starts the heartbeat loop.

Usage:
    python main.py

The bot will:
1. Load configuration from config.yaml
2. Initialize all services (Database, Trading Engine, Strategy Engines)
3. Enter a heartbeat loop that executes strategies on schedule
4. Run until interrupted (Ctrl+C)

Configuration:
    All settings are defined in config.yaml, including:
    - Strategy definitions and intervals
    - Execution provider settings
    - Database connection
    - Heartbeat interval
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import compatibility fixes BEFORE any other imports
try:
    from compat import _patch_urllib3
    _patch_urllib3()
except ImportError:
    pass

from orchestrator import BotOrchestrator


def main():
    """Main entry point for the trading bot."""
    # Initialize orchestrator (loads config.yaml by default)
    bot = BotOrchestrator()
    
    # Start the heartbeat loop
    bot.start()


if __name__ == "__main__":
    main()

