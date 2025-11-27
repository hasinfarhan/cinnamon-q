"""
Semicovariance Calculator - The Downside Risk Manager

This module calculates the Semicovariance Matrix (Downside Risk) for portfolio optimization.
Unlike standard covariance, semicovariance only considers returns below a benchmark
(e.g., 0%), ignoring "good" volatility (upside moves).

This is critical for Sortino Ratio optimization, which penalizes only downside risk.
"""

from typing import Optional

import numpy as np
import pandas as pd
from pypfopt import risk_models


class SemicovarianceCalculator:
    """
    Calculates semicovariance matrices for Sortino Ratio optimization.
    
    This is the "Downside Risk Manager" in the Bicameral Brain architecture.
    It provides downside risk metrics (semicovariance) that the optimizer uses
    to construct portfolios that maximize Sortino Ratio (return per unit of downside risk).
    
    Key Difference from Covariance:
    - Covariance: Considers ALL volatility (up and down)
    - Semicovariance: Only considers downside volatility (returns below benchmark)
    
    This aligns with Halal risk-averse principles: we want to avoid losses,
    but we don't want to penalize gains.
    
    Attributes:
        lookback_days: Number of days of history to use for semicovariance calculation
        benchmark_rate: Benchmark return rate (default: 0.0 = only negative returns count)
    """
    
    def __init__(self, lookback_days: int = 90, benchmark_rate: float = 0.0):
        """
        Initialize the semicovariance calculator.
        
        Args:
            lookback_days: Number of days of historical data to use for
                          semicovariance matrix calculation. Default: 90 days.
            benchmark_rate: Benchmark return rate. Returns below this benchmark
                           are considered "downside" and included in the matrix.
                           Default: 0.0 (only negative returns count as downside).
        """
        self.lookback_days = lookback_days
        self.benchmark_rate = benchmark_rate
    
    def get_risk_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the semicovariance matrix (downside risk matrix) from price data.
        
        This method:
        1. Filters data to the last `lookback_days`
        2. Calculates percentage returns
        3. Calculates semicovariance (only returns below benchmark)
        4. Returns the downside risk matrix
        
        Args:
            prices: DataFrame with price data.
                    Expected format:
                    - Index: Datetime (timestamps)
                    - Columns: Ticker symbols (e.g., 'BTC', 'ETH', 'SOL')
                    - Values: Close prices
        
        Returns:
            DataFrame representing the semicovariance matrix (downside risk matrix).
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
        # Use fill_method=None to avoid FutureWarning (we'll handle NaNs ourselves)
        returns = prices_filtered.pct_change(fill_method=None).dropna()
        
        if returns.empty:
            raise ValueError("No valid returns calculated (all NaN after pct_change)")
        
        if len(returns) < 2:
            raise ValueError(f"Insufficient returns data: need at least 2 rows, got {len(returns)}")
        
        # Calculate semicovariance matrix
        # Semicovariance only considers returns below the benchmark (downside risk)
        # This is what we need for Sortino Ratio optimization
        try:
            # PyPortfolioOpt's semicovariance function calculates the downside risk matrix
            # It only considers returns below the benchmark_rate (default: 0.0)
            # Returns below benchmark are "bad" volatility, returns above are "good" volatility
            semicov_matrix = risk_models.semicovariance(
                returns,
                benchmark=self.benchmark_rate
            )
        except Exception as e:
            # Fallback to sample semicovariance if PyPortfolioOpt fails
            print(f"  ⚠ Warning: PyPortfolioOpt semicovariance failed ({e}), using manual calculation")
            semicov_matrix = self._calculate_semicovariance_manual(returns)
        
        # Ensure the semicovariance matrix is a DataFrame with proper index/columns
        if not isinstance(semicov_matrix, pd.DataFrame):
            semicov_matrix = pd.DataFrame(
                semicov_matrix,
                index=returns.columns,
                columns=returns.columns
            )
        
        # Add regularization to prevent numerical issues (non-convex optimization errors)
        # Add a small identity matrix to ensure positive definiteness
        # This prevents "The problem seems to be non-convex" errors in the optimizer
        regularization = 1e-6  # Small regularization factor
        identity = pd.DataFrame(
            np.eye(len(semicov_matrix)),
            index=semicov_matrix.index,
            columns=semicov_matrix.columns
        )
        semicov_matrix = semicov_matrix + (regularization * identity)
        
        return semicov_matrix
    
    def _calculate_semicovariance_manual(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Manual calculation of semicovariance matrix (fallback method).
        
        This calculates the covariance of returns that are below the benchmark,
        ignoring returns above the benchmark.
        
        Args:
            returns: DataFrame of percentage returns
        
        Returns:
            DataFrame representing the semicovariance matrix
        """
        # Create a mask for downside returns (returns below benchmark)
        downside_mask = returns < self.benchmark_rate
        
        # Initialize semicovariance matrix
        n_assets = len(returns.columns)
        semicov_matrix = np.zeros((n_assets, n_assets))
        
        # Calculate semicovariance for each pair of assets
        for i, asset_i in enumerate(returns.columns):
            for j, asset_j in enumerate(returns.columns):
                # Get downside returns for both assets
                # Only consider periods where at least one asset had downside return
                combined_mask = downside_mask[asset_i] | downside_mask[asset_j]
                
                if combined_mask.sum() < 2:
                    # Not enough downside periods, use zero covariance
                    semicov_matrix[i, j] = 0.0
                else:
                    # Calculate covariance of downside returns
                    downside_i = returns.loc[combined_mask, asset_i] - self.benchmark_rate
                    downside_j = returns.loc[combined_mask, asset_j] - self.benchmark_rate
                    semicov_matrix[i, j] = downside_i.cov(downside_j)
        
        # Convert to DataFrame
        return pd.DataFrame(
            semicov_matrix,
            index=returns.columns,
            columns=returns.columns
        )

