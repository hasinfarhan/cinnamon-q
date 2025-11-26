"""
Mean Reversion Strategy - The Contrarian Benchmark

This strategy implements "Buy the Dip, Sell the Rip" logic:
- Buy assets that dropped significantly (oversold)
- Sell assets that rose significantly (overbought)

Why: Tests if AI handles volatility better than simple mean reversion math.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from execution.models import TargetPortfolio


class MeanReversionStrategy:
    """
    Mean reversion strategy: buy dips, sell rips.
    
    This strategy:
    - Calculates 24h price change for all assets
    - Buys assets that dropped > buy_threshold (e.g., -5%)
    - Sells assets that rose > sell_threshold (e.g., +5%)
    - Equal weights among selected assets
    
    Attributes:
        data_path: Path to data file
        buy_threshold: Percentage drop threshold for buying (negative, e.g., -0.05)
        sell_threshold: Percentage rise threshold for selling (positive, e.g., 0.05)
        lookback_hours: Number of hours to look back for price change (default: 24)
    """
    
    def __init__(
        self,
        data_path: Path,
        buy_threshold: float = -0.03,  # Reduced from -0.05 to -3% (more sensitive)
        sell_threshold: float = 0.03,   # Reduced from 0.05 to +3% (more sensitive)
        lookback_hours: int = 24
    ):
        """
        Initialize the Mean Reversion strategy.
        
        Args:
            data_path: Path to data file (parquet)
            buy_threshold: Percentage drop to trigger buy (negative, e.g., -0.05 = -5%)
            sell_threshold: Percentage rise to trigger sell (positive, e.g., 0.05 = +5%)
            lookback_hours: Hours to look back for price change calculation
        """
        self.data_path = Path(data_path)
        self.buy_threshold = buy_threshold  # Negative (e.g., -0.05)
        self.sell_threshold = sell_threshold  # Positive (e.g., 0.05)
        self.lookback_hours = lookback_hours
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"✓ Mean Reversion Strategy initialized")
        print(f"  Buy threshold: {buy_threshold*100:.1f}% (buy when price drops)")
        print(f"  Sell threshold: {sell_threshold*100:.1f}% (sell when price rises)")
        print(f"  Lookback: {lookback_hours} hours")
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio based on mean reversion signals.
        
        Logic:
        1. Load recent data (last lookback_hours)
        2. Calculate percentage change for each asset
        3. Select assets where pct_change < buy_threshold (dips)
        4. Exclude assets where pct_change > sell_threshold (rips)
        5. Equal weight among selected assets
        
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
        cutoff_time = latest_time - pd.Timedelta(hours=self.lookback_hours)
        
        # Filter to recent data
        recent_df = df[df['timestamp'] >= cutoff_time].copy()
        
        if recent_df.empty:
            # No recent data, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        # Calculate price change for each asset
        # Group by ticker and calculate pct_change from first to last
        price_changes = []
        
        for ticker in recent_df['ticker'].unique():
            ticker_data = recent_df[recent_df['ticker'] == ticker].sort_values('timestamp')
            
            if len(ticker_data) < 2:
                continue
            
            first_price = ticker_data.iloc[0]['close']
            last_price = ticker_data.iloc[-1]['close']
            
            if first_price > 0:
                pct_change = (last_price / first_price) - 1.0
                price_changes.append({
                    'ticker': ticker,
                    'pct_change': pct_change
                })
        
        if not price_changes:
            # No valid price changes, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        changes_df = pd.DataFrame(price_changes)
        
        # Select assets: Buy dips (pct_change < buy_threshold)
        # Exclude rips (pct_change > sell_threshold)
        buy_signals = changes_df[
            (changes_df['pct_change'] < self.buy_threshold) &
            (changes_df['pct_change'] <= self.sell_threshold)  # Not a rip
        ]
        
        if len(buy_signals) == 0:
            # No buy signals, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        # Equal weight among selected assets
        selected_assets = buy_signals['ticker'].tolist()
        weight_per_asset = 1.0 / len(selected_assets)
        
        # Create weights dictionary
        weights = {asset: weight_per_asset for asset in selected_assets}
        weights["CASH"] = 0.0  # Fully invested
        
        return TargetPortfolio(weights=weights)

