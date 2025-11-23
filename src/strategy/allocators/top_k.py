"""
Top-K Equal Weight Allocator

This module implements an allocator that selects the top K assets by signal
score and assigns equal weights to each, enforcing Halal constraints.
"""

import pandas as pd

from execution.models import TargetPortfolio
from ..interfaces import Allocator


class TopKEqualWeight(Allocator):
    """
    Top-K Equal Weight allocator.
    
    This allocator:
    1. Filters out negative scores (Long Only constraint)
    2. Selects the top K assets by signal strength
    3. Assigns equal weights to selected assets
    4. Enforces No Leverage constraint (sum of weights <= 1.0)
    
    Attributes:
        k: Maximum number of assets to hold in the portfolio
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize the TopKEqualWeight allocator.
        
        Args:
            k: Maximum number of assets to hold in the portfolio.
                Default is 3 assets.
                
        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
    
    def allocate(self, signals: pd.Series) -> TargetPortfolio:
        """
        Allocate capital based on signal scores.
        
        This method enforces Halal constraints:
        - Long Only: Filters out negative scores (score <= 0)
        - No Leverage: Sum of weights will always be <= 1.0 (equal weighting)
        
        Args:
            signals: Series with ticker symbols as index and scores as values.
                Scores typically range from -1.0 to +1.0.
                
        Returns:
            TargetPortfolio: Validated Pydantic model containing target weights.
                - If no assets have positive momentum: Returns empty weights (100% Cash/Safety)
                - Otherwise: Returns equal-weighted portfolio of top K assets
                
        Example:
            Input signals: pd.Series({'AAPL': 0.05, 'MSFT': -0.02, 'GOOGL': 0.03})
            With k=2: Returns TargetPortfolio(weights={'AAPL': 0.5, 'GOOGL': 0.5})
        """
        # Step 1: Halal Filter - Long Only constraint
        # Filter out any ticker with score <= 0 (we only buy assets with positive momentum)
        positive_signals = signals[signals > 0]
        
        # Edge Case: Bear Market - All signals are negative or zero
        # Return empty TargetPortfolio (100% Cash/Safety)
        # This triggers the Execution Engine to sell all positions
        if len(positive_signals) == 0:
            return TargetPortfolio(weights={})
        
        # Step 2: Ranking - Sort by signal value descending (strongest momentum first)
        sorted_signals = positive_signals.sort_values(ascending=False)
        
        # Step 3: Selection - Take the top k tickers
        selected_tickers = sorted_signals.head(self.k)
        
        # Step 4: Sizing - Calculate equal weight
        # weight = 1.0 / len(selected_tickers)
        # Example: If 2 tickers selected, each gets 0.5 (50%)
        # Example: If 3 tickers selected, each gets 0.333... (33.33%)
        num_selected = len(selected_tickers)
        weight = 1.0 / num_selected
        
        # Step 5: Construct weights dictionary
        # Create {ticker: weight} mapping for selected tickers
        weights_dict = {ticker: weight for ticker in selected_tickers.index}
        
        # Step 6: Return validated TargetPortfolio
        # Pydantic will validate:
        # - No Leverage: Sum of weights <= 1.0 (guaranteed by equal weighting)
        # - Long Only: All weights >= 0 (guaranteed by filtering score > 0)
        return TargetPortfolio(weights=weights_dict)

