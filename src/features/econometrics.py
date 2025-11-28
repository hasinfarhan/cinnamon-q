"""
Econometrics Features - Predictive Risk Modeling

This module provides econometric models for generating predictive features,
specifically GARCH(1,1) volatility forecasting.

Phase 6: The World-Class Upgrade - Sprint 1

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
volatility as a time-varying process, allowing the AI to "brace for impact"
before price moves occur, rather than reacting to past volatility.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Suppress warnings from arch library during fitting
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from arch import arch_model
except ImportError:
    arch_model = None
    print("⚠ Warning: 'arch' library not installed. GARCH features will not be available.")
    print("  Install with: pip install arch")


class Econometrics:
    """
    Econometric feature engineering for predictive risk modeling.
    
    This class provides methods to generate forward-looking features using
    econometric models like GARCH for volatility forecasting.
    """
    
    def __init__(self, rolling_window: int = 1000):
        """
        Initialize the Econometrics feature engineer.
        
        Args:
            rolling_window: Size of rolling window for GARCH fitting (default: 1000)
                          Larger windows = more stable but slower
                          Smaller windows = faster but less stable
        """
        self.rolling_window = rolling_window
        
        if arch_model is None:
            raise ImportError(
                "The 'arch' library is required for GARCH features. "
                "Install with: pip install arch"
            )
    
    def add_volatility_forecast(
        self,
        df: pd.DataFrame,
        ticker_col: str = 'ticker',
        price_col: str = 'close',
        output_col: str = 'garch_volatility'
    ) -> pd.DataFrame:
        """
        Add GARCH(1,1) volatility forecasts to the dataframe.
        
        This method calculates predictive volatility using GARCH modeling,
        which forecasts future volatility based on past shocks and volatility.
        
        Strategy: Uses expanding window approach for speed (fits model once per ticker
        on full history, then extracts conditional volatility for each timestep).
        
        Args:
            df: DataFrame with market data (must have ticker_col and price_col)
            ticker_col: Column name for asset ticker (default: 'ticker')
            price_col: Column name for price data (default: 'close')
            output_col: Column name for GARCH volatility output (default: 'garch_volatility')
        
        Returns:
            DataFrame with added 'garch_volatility' column
        
        Raises:
            ImportError: If arch library is not installed
            ValueError: If required columns are missing
        """
        if arch_model is None:
            raise ImportError(
                "The 'arch' library is required for GARCH features. "
                "Install with: pip install arch"
            )
        
        # Validate required columns
        if ticker_col not in df.columns:
            raise ValueError(f"Column '{ticker_col}' not found in dataframe")
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in dataframe")
        
        # Ensure timestamp is sorted for proper time series analysis
        if 'timestamp' in df.columns:
            df = df.sort_values(['timestamp', ticker_col]).reset_index(drop=True)
        else:
            df = df.sort_values(ticker_col).reset_index(drop=True)
        
        # Initialize output column
        df[output_col] = np.nan
        
        # Process each ticker separately
        unique_tickers = df[ticker_col].unique()
        print(f"📊 Calculating GARCH volatility for {len(unique_tickers)} assets...")
        
        for ticker_idx, ticker in enumerate(unique_tickers, 1):
            ticker_data = df[df[ticker_col] == ticker].copy()
            
            if len(ticker_data) < 50:  # Need minimum data for GARCH
                print(f"  ⚠ {ticker}: Insufficient data ({len(ticker_data)} rows), skipping")
                continue
            
            # Calculate log returns (required for GARCH)
            prices = ticker_data[price_col].values
            log_returns = np.diff(np.log(prices))
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_returns)
            if not np.any(valid_mask):
                print(f"  ⚠ {ticker}: No valid returns, skipping")
                continue
            
            log_returns = log_returns[valid_mask]
            
            if len(log_returns) < 50:  # Need minimum returns for GARCH
                print(f"  ⚠ {ticker}: Insufficient valid returns ({len(log_returns)}), skipping")
                continue
            
            try:
                # Fit GARCH(1,1) model on full history (expanding window)
                # This is faster than rolling window and provides stable forecasts
                model = arch_model(
                    log_returns * 100,  # Scale by 100 for numerical stability
                    vol='Garch',
                    p=1,  # ARCH term
                    q=1,  # GARCH term
                    dist='normal'
                )
                
                # Fit the model (this is the expensive operation)
                fitted = model.fit(disp='off', show_warning=False)
                
                # Extract conditional volatility (sigma) for each timestep
                # The fitted model provides volatility forecasts for the training period
                conditional_vol = fitted.conditional_volatility / 100.0  # Rescale back
                
                # Map conditional volatility back to original dataframe indices
                # The first return corresponds to the second price (index 1)
                ticker_indices = ticker_data.index.values
                
                # GARCH volatility starts from index 1 (first return)
                # We need to align with the original price indices
                if len(conditional_vol) == len(log_returns):
                    # Map volatility to price indices (shift by 1 since returns are diff)
                    for i, vol in enumerate(conditional_vol):
                        if i + 1 < len(ticker_indices):
                            df.loc[ticker_indices[i + 1], output_col] = vol
                    
                    # Forward fill the first value (no return before first price)
                    if len(ticker_indices) > 1:
                        first_vol = conditional_vol[0] if len(conditional_vol) > 0 else np.nan
                        if not np.isnan(first_vol):
                            df.loc[ticker_indices[0], output_col] = first_vol
                
                if ticker_idx % 5 == 0 or ticker_idx == len(unique_tickers):
                    print(f"  ✓ {ticker}: GARCH volatility calculated ({len(conditional_vol)} forecasts)")
            
            except Exception as e:
                print(f"  ✗ {ticker}: GARCH fitting failed ({e}), using NaN")
                # Leave as NaN for this ticker
        
        # Fill remaining NaNs with forward fill, then backward fill
        df[output_col] = df.groupby(ticker_col)[output_col].transform(
            lambda x: x.ffill().bfill()
        )
        
        # If still NaN, fill with median volatility across all assets
        if df[output_col].isna().any():
            median_vol = df[output_col].median()
            if not np.isnan(median_vol):
                df[output_col] = df[output_col].fillna(median_vol)
            else:
                # Last resort: fill with small default value
                df[output_col] = df[output_col].fillna(0.01)
        
        print(f"✓ GARCH volatility added: {df[output_col].notna().sum()} values calculated")
        print(f"  Mean volatility: {df[output_col].mean():.4f}, Std: {df[output_col].std():.4f}")
        
        return df

