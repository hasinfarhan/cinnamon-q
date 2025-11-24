"""
Covariance Calculator - The Quant Risk Manager

This module calculates the covariance matrix for portfolio optimization.
It uses Ledoit-Wolf shrinkage for robust estimation, which is critical
for noisy crypto data with limited history.
"""

from typing import Optional

import numpy as np
import pandas as pd
from pypfopt import risk_models


class CovarianceCalculator:
    """
    Calculates covariance matrices for portfolio optimization.
    
    This is the "Safety Officer" in the Bicameral Brain architecture.
    It provides risk metrics (volatility and correlation) that the
    optimizer uses to construct safe portfolios.
    
    Attributes:
        lookback_days: Number of days of history to use for covariance calculation
    """
    
    def __init__(self, lookback_days: int = 90):
        """
        Initialize the covariance calculator.
        
        Args:
            lookback_days: Number of days of historical data to use for
                          covariance matrix calculation. Default: 90 days.
        """
        self.lookback_days = lookback_days
    
    def get_covariance_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the covariance matrix from price data.
        
        This method:
        1. Filters data to the last `lookback_days`
        2. Calculates percentage returns
        3. Applies Ledoit-Wolf shrinkage for robust estimation
        4. Returns the covariance matrix
        
        Args:
            prices: DataFrame with price data.
                    Expected format:
                    - Index: Datetime (timestamps)
                    - Columns: Ticker symbols (e.g., 'BTC', 'ETH', 'SOL')
                    - Values: Close prices
        
        Returns:
            DataFrame representing the covariance matrix.
            Shape: (n_assets, n_assets)
            Index and columns: Ticker symbols
        
        Raises:
            ValueError: If prices DataFrame is empty or has insufficient data
        """
        if prices.empty:
            raise ValueError("Prices DataFrame is empty")
        
        if len(prices) < 2:
            raise ValueError(f"Insufficient data: need at least 2 rows, got {len(prices)}")
        
        # Filter to last lookback_days
        # If we have less data than lookback_days, use all available data
        if len(prices) > self.lookback_days:
            prices_filtered = prices.tail(self.lookback_days)
        else:
            prices_filtered = prices
            print(f"  ⚠ Warning: Only {len(prices)} days available, using all data (requested {self.lookback_days})")
        
        # Ensure prices are sorted by date (ascending)
        prices_filtered = prices_filtered.sort_index()
        
        # Calculate percentage returns
        # pct_change() calculates (price_t - price_{t-1}) / price_{t-1}
        returns = prices_filtered.pct_change().dropna()
        
        if returns.empty:
            raise ValueError("No valid returns calculated (all NaN after pct_change)")
        
        if len(returns) < 2:
            raise ValueError(f"Insufficient returns data: need at least 2 rows, got {len(returns)}")
        
        # Use Ledoit-Wolf shrinkage for robust covariance estimation
        # This is more robust than simple df.cov() for noisy crypto data
        # Ledoit-Wolf shrinks the sample covariance matrix towards a structured estimator
        # This reduces estimation error, especially important for:
        # - Small sample sizes (crypto has limited history)
        # - High-dimensional problems (many assets)
        # - Noisy data (crypto is volatile)
        try:
            cov_matrix = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
        except Exception as e:
            # Fallback to sample covariance if Ledoit-Wolf fails
            print(f"  ⚠ Warning: Ledoit-Wolf shrinkage failed ({e}), using sample covariance")
            cov_matrix = returns.cov()
        
        # Ensure the covariance matrix is a DataFrame with proper index/columns
        if not isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = pd.DataFrame(
                cov_matrix,
                index=returns.columns,
                columns=returns.columns
            )
        
        return cov_matrix
    
    def get_correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix from price data.
        
        This is a convenience method that converts the covariance matrix
        to a correlation matrix. Useful for visualization and analysis.
        
        Args:
            prices: DataFrame with price data (same format as get_covariance_matrix)
        
        Returns:
            DataFrame representing the correlation matrix.
            Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).
        """
        cov_matrix = self.get_covariance_matrix(prices)
        
        # Convert covariance to correlation
        # Correlation = Cov(X,Y) / (Std(X) * Std(Y))
        std_devs = np.sqrt(np.diag(cov_matrix))
        correlation = cov_matrix / np.outer(std_devs, std_devs)
        
        # Ensure it's a DataFrame
        if not isinstance(correlation, pd.DataFrame):
            correlation = pd.DataFrame(
                correlation,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )
        
        return correlation

