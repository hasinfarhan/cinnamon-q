"""
Risk Management Module - The Quant Risk Manager

This module provides risk calculation tools for the Bicameral Brain architecture.
The Risk Manager calculates covariance matrices and other risk metrics to inform
the optimizer's portfolio construction decisions.
"""

from .covariance import CovarianceCalculator

__all__ = ["CovarianceCalculator"]

