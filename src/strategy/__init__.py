"""
Strategy Engine Module

This module provides a modular, data-driven decision engine that consumes
historical market data and outputs strictly validated TargetPortfolio objects.

The engine separates Signal Generation (Analysis) from Capital Allocation (Risk/Sizing).
"""

from .allocators.top_k import TopKEqualWeight
from .engine import StrategyEngine
from .interfaces import Allocator, SignalGenerator
from .signals.momentum import MomentumSignal

__all__ = [
    "SignalGenerator",
    "Allocator",
    "MomentumSignal",
    "TopKEqualWeight",
    "StrategyEngine"
]

