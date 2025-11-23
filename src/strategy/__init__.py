"""
Strategy Engine Module

This module provides a modular, data-driven decision engine that consumes
historical market data and outputs strictly validated TargetPortfolio objects.

The engine separates Signal Generation (Analysis) from Capital Allocation (Risk/Sizing).
"""

from .interfaces import Allocator, SignalGenerator

__all__ = ["SignalGenerator", "Allocator"]

