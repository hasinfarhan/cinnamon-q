"""
Trend Following Strategy - The Momentum Benchmark

This strategy implements Golden Cross logic:
- Buy assets where price > SMA(long_window)
- Sell assets where price < SMA(long_window)

Why: Tests if AI can catch big runs better than a simple moving average.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from execution.models import TargetPortfolio


class TrendFollowingStrategy:
    """
    Trend following strategy using Simple Moving Average (SMA).
    
    This strategy:
    - Calculates SMA(long_window) for each asset
    - Buys assets where current_price > SMA (uptrend)
    - Sells assets where current_price < SMA (downtrend)
    - Equal weights among selected assets
    
    Attributes:
        data_path: Path to data file
        window: SMA window size (default: 200)
    """
    
    def __init__(
        self,
        data_path: Path,
        window: int = 200
    ):
        """
        Initialize the Trend Following strategy.
        
        Args:
            data_path: Path to data file (parquet)
            window: SMA window size (number of periods, default: 200)
        """
        self.data_path = Path(data_path)
        self.window = window
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"✓ Trend Following Strategy initialized")
        print(f"  SMA window: {window} periods")
        print(f"  Logic: Buy when price > SMA{window}, Sell when price < SMA{window}")
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio based on trend signals.
        
        Logic:
        1. Load historical data
        2. Calculate SMA(window) for each asset
        3. Select assets where current_price > SMA (uptrend)
        4. Equal weight among selected assets
        
        Args:
            portfolio_value: Current portfolio value (unused, for compatibility)
        
        Returns:
            TargetPortfolio with weights for trending assets, rest in cash
        """
        # Load data
        df = pd.read_parquet(self.data_path)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
        
        # Calculate SMA and current price for each asset
        trend_signals = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            if len(ticker_data) < self.window:
                # Not enough data for SMA
                continue
            
            # Calculate SMA
            ticker_data['sma'] = ticker_data['close'].rolling(window=self.window, min_periods=self.window).mean()
            
            # Get latest values
            latest = ticker_data.iloc[-1]
            current_price = latest['close']
            sma_value = latest['sma']
            
            if pd.isna(sma_value):
                continue
            
            # Buy signal: price > SMA (uptrend)
            if current_price > sma_value:
                trend_signals.append(ticker)
        
        if len(trend_signals) == 0:
            # No trend signals, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        # Equal weight among selected assets
        weight_per_asset = 1.0 / len(trend_signals)
        
        # Create weights dictionary
        weights = {asset: weight_per_asset for asset in trend_signals}
        weights["CASH"] = 0.0  # Fully invested
        
        return TargetPortfolio(weights=weights)

