"""
Risk Validators - Hard Constraints for AI Strategies

This package contains deterministic validators that enforce hard risk limits,
acting as "Circuit Breakers" to override AI decisions when necessary.

Philosophy: "The AI is probabilistic (it might avoid risk). The Validator is deterministic (it will avoid risk)."
"""

from .risk import CalmarRiskGate
from .macro import MacroRegimeGate

__all__ = [
    "CalmarRiskGate",
    "MacroRegimeGate"
]

