"""
Portfolio Optimizer Module - The Negotiator

This module provides portfolio optimization tools for the Bicameral Brain architecture.
The Optimizer takes ML signals (expected returns) and risk metrics (covariance or semicovariance)
to solve for optimal portfolio weights using mean-variance optimization.

Two optimizers are available:
- MeanVarianceOptimizer: Maximizes Sharpe Ratio (uses covariance matrix)
- SortinoRatioOptimizer: Maximizes Sortino Ratio (uses semicovariance matrix)
"""

from .mean_variance import MeanVarianceOptimizer
from .sortino_solver import SortinoRatioOptimizer

__all__ = ["MeanVarianceOptimizer", "SortinoRatioOptimizer"]

