"""
Signal Generator Implementations

This module contains concrete implementations of the SignalGenerator interface.
"""

from .llm_crypto import LLMCryptoSignal
from .momentum import MomentumSignal

__all__ = ["MomentumSignal", "LLMCryptoSignal"]

