"""
Abstract interface for execution providers.

This module defines the contract that all broker adapters must implement,
ensuring broker-agnostic execution capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict

from .models import PortfolioState, TradeOrder


class ExecutionProvider(ABC):
    """
    Abstract base class for execution providers.
    
    All broker-specific implementations must inherit from this class
    and implement all abstract methods.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Authenticate and establish connection with the broker.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_state(self) -> PortfolioState:
        """
        Get the current portfolio state from the broker.
        
        Returns:
            PortfolioState: Validated Pydantic model containing:
                - cash: Available cash balance
                - total_equity: Total portfolio value
                - positions: Dictionary mapping ticker symbols to quantities
                - timestamp: When the snapshot was taken
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions from the broker.
        
        DEPRECATED: Use get_state().positions instead.
        This method is kept for backward compatibility.
        
        Returns:
            Dict[str, float]: Dictionary mapping ticker symbols to quantities.
                Positive values indicate long positions, negative values indicate short positions.
        """
        pass
    
    @abstractmethod
    def get_cash_balance(self) -> float:
        """
        Get the total available cash balance.
        
        DEPRECATED: Use get_state().cash instead.
        This method is kept for backward compatibility.
        
        Returns:
            float: Available cash balance in USD
        """
        pass
    
    @abstractmethod
    def execute_order(self, order: TradeOrder) -> dict:
        """
        Execute a trade order with the broker.
        
        Args:
            order: The TradeOrder object containing order details
            
        Returns:
            dict: Raw response from the broker API
            
        Raises:
            ValueError: If order validation fails
            Exception: If order execution fails
        """
        pass


