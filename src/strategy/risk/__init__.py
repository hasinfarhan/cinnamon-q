"""
Risk Management Module - The Quant Risk Manager

This module provides risk calculation tools for the Bicameral Brain architecture.
The Risk Manager calculates risk matrices (covariance or semicovariance) to inform
the optimizer's portfolio construction decisions.
"""

from .covariance import CovarianceCalculator
from .semicovariance import SemicovarianceCalculator

__all__ = ["CovarianceCalculator", "SemicovarianceCalculator"]

