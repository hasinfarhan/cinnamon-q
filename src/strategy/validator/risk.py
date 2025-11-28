"""
Calmar Risk Gate - Hard Drawdown Limit Validator

This module implements a "Circuit Breaker" that forces a retreat to 100% cash
if the portfolio's drawdown exceeds a specified limit. This acts as a hard
constraint that overrides AI decisions when necessary.

Phase 5.6 Sprint 2: The Calmar Gate

Philosophy: "The AI is probabilistic (it might avoid risk). The Validator is deterministic (it will avoid risk)."
"""

import logging
from typing import Optional

from execution.models import PortfolioState, TargetPortfolio

logger = logging.getLogger(__name__)


class CalmarRiskGate:
    """
    Hard-coded Risk Validator that acts as a "Circuit Breaker."
    
    If the portfolio's drawdown exceeds a limit, it forces a retreat to Cash,
    overriding the AI's target portfolio.
    
    Features:
    - Tracks peak equity (high water mark) across all validations
    - Locks when drawdown exceeds max_drawdown_limit (default: 15%)
    - Unlocks when drawdown falls below unlock_threshold (default: 10%)
    - Hysteresis loop prevents flickering (lock/unlock oscillations)
    
    Example:
        gate = CalmarRiskGate(max_drawdown_limit=0.15, unlock_threshold=0.10)
        safe_target = gate.validate(ai_target, current_portfolio_state)
    """
    
    def __init__(
        self,
        max_drawdown_limit: float = 0.15,  # 15% drawdown triggers lock
        unlock_threshold: float = 0.10  # 10% drawdown required to unlock
    ):
        """
        Initialize the Calmar Risk Gate.
        
        Args:
            max_drawdown_limit: Maximum allowed drawdown (0.15 = 15%). If exceeded, gate locks.
            unlock_threshold: Drawdown threshold to unlock gate (0.10 = 10%). Must be < max_drawdown_limit.
        
        Raises:
            ValueError: If unlock_threshold >= max_drawdown_limit
        """
        if unlock_threshold >= max_drawdown_limit:
            raise ValueError(
                f"unlock_threshold ({unlock_threshold}) must be < max_drawdown_limit ({max_drawdown_limit}) "
                "to create a hysteresis loop and prevent flickering."
            )
        
        self.max_drawdown_limit = max_drawdown_limit
        self.unlock_threshold = unlock_threshold
        
        # State tracking
        self.peak_equity: float = 0.0  # High water mark (never decreases)
        self.is_locked: bool = False  # Gate lock status
        
        logger.info(
            f"🛡️ Calmar Risk Gate initialized: "
            f"Max Drawdown: {max_drawdown_limit*100:.1f}%, "
            f"Unlock Threshold: {unlock_threshold*100:.1f}%"
        )
    
    def validate(
        self,
        target: TargetPortfolio,
        current_state: PortfolioState
    ) -> TargetPortfolio:
        """
        Validate and potentially override the target portfolio based on drawdown.
        
        This method:
        1. Updates peak equity (high water mark)
        2. Calculates current drawdown
        3. Checks if gate should lock/unlock
        4. Returns original target or 100% cash override
        
        Args:
            target: The AI's target portfolio (may be overridden)
            current_state: Current portfolio state (for equity calculation)
        
        Returns:
            TargetPortfolio: Either the original target or 100% cash if gate is locked
        """
        # Step 1: Update Metrics
        current_equity = float(current_state.total_equity)
        
        # Update peak equity (high water mark - never decreases)
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Calculate current drawdown
        if self.peak_equity > 1e-8:  # Avoid division by zero
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            # Ensure drawdown is non-negative (should never be negative, but safety check)
            drawdown = max(0.0, drawdown)
        else:
            drawdown = 0.0
        
        # Step 2: Check Logic (Hysteresis Loop)
        # Lock if drawdown exceeds limit
        if drawdown > self.max_drawdown_limit:
            if not self.is_locked:
                self.is_locked = True
                logger.warning(
                    f"🚨 RISK ALERT: Drawdown {drawdown*100:.2f}% exceeds limit {self.max_drawdown_limit*100:.1f}%. "
                    f"Fortress Gate LOCKED. Forcing 100% cash."
                )
        
        # Unlock if drawdown falls below threshold (hysteresis prevents flickering)
        elif self.is_locked and drawdown < self.unlock_threshold:
            self.is_locked = False
            logger.info(
                f"✅ RISK RECOVERY: Drawdown {drawdown*100:.2f}% below threshold {self.unlock_threshold*100:.1f}%. "
                f"Fortress Gate UNLOCKED. Resuming normal operations."
            )
        
        # Step 3: Enforce
        if self.is_locked:
            # Return 100% cash override (safety mode)
            return TargetPortfolio(weights={'CASH': 1.0})
        else:
            # Return original target (AI decision stands)
            return target
    
    def reset(self):
        """
        Reset the gate state (useful for testing or new episodes).
        
        This resets peak equity and lock status, but keeps the limits unchanged.
        """
        self.peak_equity = 0.0
        self.is_locked = False
        logger.debug("🔄 Calmar Risk Gate reset")

