"""
Abstract interfaces for the Strategy Engine.

This module defines the contracts that all signal generators and allocators
must implement, ensuring a consistent pipeline architecture.
"""

from abc import ABC, abstractmethod

import pandas as pd

# Import from execution module (sibling package)
# Note: Assumes 'src' is in Python path (standard project structure)
from execution.models import TargetPortfolio


class SignalGenerator(ABC):
    """
    Abstract base class for signal generators.
    
    Signal generators are pure analysis components that examine market data
    (price, volume, news, etc.) and assign a score to each asset.
    
    The score typically ranges from -1.0 (strong sell) to +1.0 (strong buy),
    where 0.0 represents neutral.
    """
    
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
                Expected to have a MultiIndex or columns that allow identification
                of ticker symbols and dates.
                
        Returns:
            pd.Series: Series with ticker symbols as index and scores as values.
                Scores typically range from -1.0 (strong sell) to +1.0 (strong buy).
                Example: pd.Series({'AAPL': 0.75, 'MSFT': 0.50, 'GOOGL': -0.25})
        """
        pass


class Allocator(ABC):
    """
    Abstract base class for portfolio allocators.
    
    Allocators convert abstract signal scores into concrete portfolio weights.
    They handle risk management, position sizing, and enforce Halal constraints
    (Long Only, No Leverage).
    """
    
    @abstractmethod
    def allocate(self, signals: pd.Series) -> TargetPortfolio:
        """
        Allocate capital based on signal scores.
        
        Args:
            signals: Series with ticker symbols as index and scores as values.
                Scores typically range from -1.0 to +1.0.
                
        Returns:
            TargetPortfolio: Validated Pydantic model containing target weights.
                Must satisfy Halal constraints:
                - Long Only: All weights >= 0
                - No Leverage: Sum of weights <= 1.0
                
        Raises:
            ValueError: If allocation violates constraints (handled by TargetPortfolio validation)
        """
        pass

