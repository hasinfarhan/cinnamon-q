"""
Portfolio Optimizer Module - The Negotiator

This module provides portfolio optimization tools for the Bicameral Brain architecture.
The Optimizer takes ML signals (expected returns) and risk metrics (covariance) to
solve for optimal portfolio weights using mean-variance optimization.
"""

from .mean_variance import MeanVarianceOptimizer

__all__ = ["MeanVarianceOptimizer"]

