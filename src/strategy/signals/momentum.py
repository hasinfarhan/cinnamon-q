"""
Momentum Signal Generator

This module implements a momentum-based signal generator that calculates
the percentage price change over a specified time window to identify
trending assets.
"""

import pandas as pd

from ..interfaces import SignalGenerator


class MomentumSignal(SignalGenerator):
    """
    Momentum-based signal generator.
    
    Calculates the percentage change in closing price over a specified
    time window to identify assets that are trending up (positive score)
    or trending down (negative score).
    
    Attributes:
        window_days: Number of days to look back for momentum calculation
    """
    
    def __init__(self, window_days: int = 30):
        """
        Initialize the MomentumSignal generator.
        
        Args:
            window_days: Number of days to look back for momentum calculation.
                Default is 30 days.
        """
        if window_days <= 0:
            raise ValueError("window_days must be positive")
        self.window_days = window_days
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals from price data.
        
        Args:
            data: DataFrame with columns ['date', 'ticker', 'close'].
                Expected structure:
                - 'date': Date/timestamp column
                - 'ticker': Ticker symbol column
                - 'close': Closing price column
                
        Returns:
            pd.Series: Series with ticker symbols as index and momentum scores as values.
                Scores represent percentage change over the window period.
                Example: 0.05 means 5% gain, -0.03 means 3% loss.
                NaN values (insufficient history) are filled with 0.0 (neutral).
        """
        # Validate required columns
        required_columns = ['date', 'ticker', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure data is sorted by date (critical for correct pct_change calculation)
        data_sorted = data.sort_values('date').copy()
        
        # Group by ticker FIRST, then calculate percentage change
        # This is critical: without grouping, pct_change would mix prices
        # between different assets (e.g., Apple's price vs Microsoft's price)
        grouped = data_sorted.groupby('ticker')['close']
        
        # Calculate percentage change over window_days
        # pct_change(periods=window_days) calculates: (price_today - price_N_days_ago) / price_N_days_ago
        momentum = grouped.pct_change(periods=self.window_days)
        
        # Extract the most recent (last) momentum value for each ticker
        # The momentum Series has the same index as data_sorted, so we can group by ticker
        # from the original dataframe to get the last value per ticker
        latest_momentum = momentum.groupby(data_sorted['ticker']).last()
        
        # Fill NaN values with 0.0 (neutral signal)
        # NaN occurs when:
        # 1. Insufficient history (< window_days)
        # 2. Missing data for a ticker
        latest_momentum = latest_momentum.fillna(0.0)
        
        # Clean up Series name for clarity
        latest_momentum.name = None
        
        return latest_momentum

