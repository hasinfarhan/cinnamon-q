"""
Volatility Grid Strategy - The "Chop" Master

This benchmark simulates a grid trading bot that uses Bollinger Bands to
identify oversold (buy) and overbought (sell) conditions across all assets.

Phase 5.5 Sprint 3: Advanced Benchmark Suite

Logic:
- Calculate Bollinger Bands for all assets
- Score assets based on position relative to bands:
  - Close < Lower Band: Score = 2 (Strong Buy)
  - Lower < Close < SMA: Score = 1 (Weak Buy)
  - Close > SMA: Score = 0 (Sell/Cash)
- Normalize scores to sum to 1.0
- If all scores = 0, go 100% cash
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from execution.models import TargetPortfolio
from strategy.ppo_strategy import load_asset_universe


class VolatilityGridStrategy:
    """
    A grid trading strategy based on Bollinger Bands.
    
    This strategy is designed to profit from sideways/choppy markets by
    buying at support (lower band) and selling at resistance (upper band).
    
    Logic:
    - Calculate Bollinger Bands (SMA ± N*STD) for each asset
    - Score assets: 2 (strong buy), 1 (weak buy), 0 (sell/cash)
    - Allocate weights proportional to scores
    - If no buy signals, go 100% cash
    """
    
    def __init__(
        self,
        data_path: Path,
        universe_csv: Optional[Path] = None,
        window: int = 20,  # SMA period for Bollinger Bands
        num_std_dev: float = 2.0  # Number of standard deviations for band width
    ):
        """
        Initialize the Volatility Grid strategy.
        
        Args:
            data_path: Path to data file (parquet)
            universe_csv: Path to crypto_universe.csv (default: auto-detect)
            window: The lookback window for the Simple Moving Average (SMA)
            num_std_dev: The number of standard deviations to use for the Bollinger Bands
        """
        self.data_path = Path(data_path)
        self.window = window
        self.num_std_dev = num_std_dev
        self.asset_list = load_asset_universe(universe_csv)
        
        if not self.asset_list:
            # Fallback to assets from data_path
            try:
                df_sample = pd.read_parquet(self.data_path, columns=['ticker'])
                self.asset_list = sorted(df_sample['ticker'].unique().tolist())
                # Ensure Alpaca format
                self.asset_list = [f"{a}/USD" if '/' not in a and not a.endswith('USD') else a for a in self.asset_list]
                self.asset_list = sorted(self.asset_list)
            except Exception:
                raise ValueError("Could not determine asset list from universe CSV or data_path.")
        
        self.n_assets = len(self.asset_list)
        
        print(f"✓ Volatility Grid Strategy initialized")
        print(f"  SMA window: {window} periods")
        print(f"  Std Dev: {num_std_dev}")
        print(f"  Logic: Score = 2 (Close < Lower), 1 (Lower < Close < SMA), 0 (Close > SMA)")
        print(f"  Assets: {self.n_assets}")
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio based on Bollinger Band grid signals.
        
        Logic:
        1. Fetch latest data for all assets.
        2. Calculate SMA, STD, Upper Band, and Lower Band for each asset.
        3. Score each asset:
           - Score = 2 if Close < Lower Band (strong buy)
           - Score = 1 if Lower Band < Close < SMA (weak buy)
           - Score = 0 if Close > SMA (sell/cash)
        4. Normalize scores to sum to 1.0.
        5. If all scores = 0, go 100% cash.
        
        Args:
            portfolio_value: Current portfolio value (not used for allocation logic)
        
        Returns:
            TargetPortfolio with weights proportional to grid scores.
        """
        # Load full data
        df_full = pd.read_parquet(self.data_path)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_full['timestamp']):
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        
        # Get the latest timestamp from the data
        latest_timestamp = df_full['timestamp'].max()
        
        # Filter data to ensure enough history for Bollinger Band calculation
        df_history = df_full[df_full['timestamp'] <= latest_timestamp].copy()
        
        if df_history.empty:
            print(f"  ⚠ No historical data for Bollinger Bands, going 100% cash.")
            return TargetPortfolio(weights={'CASH': 1.0})
        
        # Calculate scores for each asset
        asset_scores = {}
        
        for asset in self.asset_list:
            base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
            asset_data = df_history[df_history['ticker'] == base_asset].sort_values('timestamp')
            
            if len(asset_data) < self.window:
                # Not enough data, skip
                asset_scores[base_asset] = 0
                continue
            
            # Calculate Bollinger Bands
            sma = asset_data['close'].rolling(window=self.window, min_periods=1).mean()
            std = asset_data['close'].rolling(window=self.window, min_periods=1).std()
            
            upper_band = sma + (std * self.num_std_dev)
            lower_band = sma - (std * self.num_std_dev)
            
            # Get latest values
            latest_close = asset_data['close'].iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            latest_sma = sma.iloc[-1]
            
            # Score based on position relative to bands
            if pd.isna(latest_close) or pd.isna(latest_lower) or pd.isna(latest_sma):
                asset_scores[base_asset] = 0
                continue
            
            if latest_close < latest_lower:
                # Strong buy: Price below lower band (oversold)
                asset_scores[base_asset] = 2
            elif latest_close < latest_sma:
                # Weak buy: Price between lower band and SMA
                asset_scores[base_asset] = 1
            else:
                # Sell/Cash: Price above SMA (overbought or neutral)
                asset_scores[base_asset] = 0
        
        # Normalize scores to sum to 1.0
        total_score = sum(asset_scores.values())
        
        weights: Dict[str, float] = {}
        
        if total_score > 0:
            # Allocate weights proportional to scores
            for asset, score in asset_scores.items():
                if score > 0:
                    weights[asset] = score / total_score
            weights['CASH'] = 0.0  # Fully invested
        else:
            # No buy signals, go 100% cash
            weights['CASH'] = 1.0
        
        return TargetPortfolio(weights=weights)

