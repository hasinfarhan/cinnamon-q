"""
Benchmark Strategies - Control Groups for AI Validation

This package contains simple, rule-based trading strategies that serve as
baselines for comparing against complex AI models (PPO, LSTM, etc.).

These strategies implement the same interface as AI strategies (get_target_portfolio)
so they can be swapped in/out via config.yaml.

Philosophies:
- Buy and Hold: Passive equal-weight indexing
- Mean Reversion: Buy dips, sell rips
- Trend Following: Momentum-based (SMA crossover)
"""

from .buy_and_hold import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .trend import TrendFollowingStrategy
from .blue_chip import BlueChipStrategy
from .volatility_grid import VolatilityGridStrategy

__all__ = [
    "BuyAndHoldStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy"
]

