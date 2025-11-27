"""
Mean Reversion Strategy - The Contrarian Benchmark (Bollinger Bands Edition)

This strategy uses Bollinger Bands (SMA ± N*STD) to identify oversold/overbought
conditions. It buys assets below the lower band (oversold) and exits to cash
when assets are above the upper band (overbought).

Why: Tests if AI handles volatility better than dynamic mean reversion math.
The Bollinger Bands adapt to current market volatility, making it more responsive
than fixed percentage thresholds.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from execution.models import TargetPortfolio


class MeanReversionStrategy:
    """
    Mean reversion strategy using Bollinger Bands for dynamic volatility thresholds.
    
    This strategy:
    - Calculates Bollinger Bands (SMA ± N*STD) for each asset
    - Buys assets where current_price < Lower Band (oversold)
    - Sells assets where current_price > Upper Band (overbought) → Exit to cash
    - Equal weights among selected assets
    - Long Only: Respects Halal constraints (no shorting)
    
    Attributes:
        data_path: Path to data file
        window: SMA period for Bollinger Bands (default: 20)
        num_std_dev: Number of standard deviations for band width (default: 2.0)
    """
    
    def __init__(
        self,
        data_path: Path,
        window: int = 20,  # SMA period for Bollinger Bands
        num_std_dev: float = 2.0  # Number of standard deviations for band width
    ):
        """
        Initialize the Mean Reversion strategy with Bollinger Bands.
        
        Args:
            data_path: Path to data file (parquet)
            window: SMA period for Bollinger Bands (default: 20)
            num_std_dev: Number of standard deviations for band width (default: 2.0)
        """
        self.data_path = Path(data_path)
        self.window = window
        self.num_std_dev = num_std_dev
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"✓ Mean Reversion Strategy (Bollinger Bands) initialized")
        print(f"  SMA window: {window} periods")
        print(f"  Band width: ±{num_std_dev} standard deviations")
        print(f"  Buy signal: Price < Lower Band (oversold)")
        print(f"  Sell signal: Price > Upper Band (overbought, exit to cash)")
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio based on Bollinger Band mean reversion signals.
        
        Logic:
        1. Load data for all assets
        2. For each asset, calculate Bollinger Bands (SMA ± N*STD)
        3. Buy Signal: Current price < Lower Band (oversold)
        4. Sell Signal: Current price > Upper Band (overbought) → Exit to cash
        5. Allocate equal weight to "buy" assets. If no buy signals, go 100% cash
        
        Note: This is Long Only - we exit to cash on sell signals, we do not short.
        
        Args:
            portfolio_value: Current portfolio value (unused, for compatibility)
        
        Returns:
            TargetPortfolio with weights for selected assets, rest in cash
        """
        # Load data
        df = pd.read_parquet(self.data_path)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get latest timestamp
        latest_time = df['timestamp'].max()
        
        # Filter data to ensure we have enough history for Bollinger Bands
        # Need at least `window` periods for SMA/STD calculation
        df_history = df[df['timestamp'] <= latest_time].copy()
        
        if df_history.empty:
            # No historical data, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        buy_candidates = []
        
        # Calculate Bollinger Bands for each asset
        for ticker in df_history['ticker'].unique():
            asset_data = df_history[df_history['ticker'] == ticker].sort_values('timestamp')
            
            # Need at least `window` periods for Bollinger Bands
            if len(asset_data) < self.window:
                continue  # Skip assets with insufficient data
            
            # Calculate SMA (Simple Moving Average)
            sma = asset_data['close'].rolling(window=self.window).mean()
            
            # Calculate STD (Standard Deviation)
            std = asset_data['close'].rolling(window=self.window).std()
            
            # Calculate Bollinger Bands
            upper_band = sma + (std * self.num_std_dev)
            lower_band = sma - (std * self.num_std_dev)
            
            # Get the latest values (last row)
            latest_close = asset_data['close'].iloc[-1]
            latest_lower_band = lower_band.iloc[-1]
            latest_upper_band = upper_band.iloc[-1]
            
            # Check if we have valid band values (not NaN)
            if pd.isna(latest_lower_band) or pd.isna(latest_upper_band):
                continue  # Skip if bands are not calculated yet
            
            # Buy Signal: Current price < Lower Band (oversold - mean reversion opportunity)
            if latest_close < latest_lower_band:
                buy_candidates.append(ticker)
            # Sell Signal: Current price > Upper Band (overbought)
            # Note: We don't actively track positions here, but this logic would
            # trigger an exit to cash if we were holding this asset.
            # For simplicity, we just don't buy overbought assets.
        
        if len(buy_candidates) == 0:
            # No buy signals, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        # Equal weight among buy candidates
        weight_per_asset = 1.0 / len(buy_candidates)
        
        # Create weights dictionary
        weights = {asset: weight_per_asset for asset in buy_candidates}
        weights["CASH"] = 0.0  # Fully invested
        
        # Normalize to ensure sum <= 1.0 (Halal constraint: no leverage)
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return TargetPortfolio(weights=weights)
