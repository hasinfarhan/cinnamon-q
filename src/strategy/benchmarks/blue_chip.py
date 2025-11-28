"""
Blue Chip Strategy - The "Safe" Crypto Portfolio

This benchmark holds equal weight in only the three largest, most established
crypto assets: BTC, ETH, and SOL. This represents a conservative "blue chip"
approach to crypto investing.

Phase 5.5 Sprint 3: Advanced Benchmark Suite
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from execution.models import TargetPortfolio
from strategy.ppo_strategy import load_asset_universe


class BlueChipStrategy:
    """
    A conservative "Blue Chip" strategy that allocates equal weight to only
    BTC, ETH, and SOL - the three largest and most established crypto assets.
    
    This serves as a "safe" benchmark that avoids smaller altcoins.
    
    Logic:
    - Equal weight (1/3) for BTC, ETH, SOL
    - All other assets = 0
    - Cash = 0 (fully invested)
    """
    
    # Blue chip assets (must be in Alpaca format)
    BLUE_CHIP_ASSETS = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    
    def __init__(self, data_path: Path, universe_csv: Optional[Path] = None):
        """
        Initialize the Blue Chip strategy.
        
        Args:
            data_path: Path to data file (parquet) - used for asset list fallback
            universe_csv: Path to crypto_universe.csv (default: auto-detect)
        """
        self.data_path = Path(data_path)
        
        # Verify blue chip assets are available
        asset_list = load_asset_universe(universe_csv)
        if not asset_list:
            # Fallback to assets from data_path
            try:
                df_sample = pd.read_parquet(self.data_path, columns=['ticker'])
                asset_list = sorted(df_sample['ticker'].unique().tolist())
                # Ensure Alpaca format
                asset_list = [f"{a}/USD" if '/' not in a and not a.endswith('USD') else a for a in asset_list]
            except Exception:
                raise ValueError("Could not determine asset list from universe CSV or data_path.")
        
        # Check which blue chip assets are available
        self.available_blue_chips = []
        for asset in self.BLUE_CHIP_ASSETS:
            # Try to match asset (handle format differences)
            base_asset = asset.replace('/USD', '')
            for available_asset in asset_list:
                available_base = available_asset.replace('/USD', '')
                if base_asset == available_base:
                    self.available_blue_chips.append(asset)
                    break
        
        if not self.available_blue_chips:
            raise ValueError(
                f"None of the blue chip assets {self.BLUE_CHIP_ASSETS} are available in the data. "
                f"Available assets: {asset_list[:10]}..."
            )
        
        self.n_blue_chips = len(self.available_blue_chips)
        self.equal_weight = 1.0 / self.n_blue_chips if self.n_blue_chips > 0 else 0.0
        
        print(f"✓ Blue Chip Strategy initialized")
        print(f"  Blue chip assets: {self.available_blue_chips}")
        print(f"  Allocation: Equal weight ({self.equal_weight:.2%} per asset)")
        if self.n_blue_chips < len(self.BLUE_CHIP_ASSETS):
            missing = set(self.BLUE_CHIP_ASSETS) - set(self.available_blue_chips)
            print(f"  ⚠ Missing assets: {missing}")
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate a target portfolio with equal weights for blue chip assets only.
        
        Args:
            portfolio_value: Current portfolio value (not used for allocation logic)
        
        Returns:
            TargetPortfolio with equal weights for BTC, ETH, SOL and 0% cash.
        """
        weights: Dict[str, float] = {}
        
        # Allocate equal weight to each available blue chip asset
        for asset in self.available_blue_chips:
            # Remove /USD suffix for TargetPortfolio (uses base symbol)
            base_symbol = asset.replace('/USD', '')
            weights[base_symbol] = self.equal_weight
        
        # Fully invested (no cash)
        weights['CASH'] = 0.0
        
        return TargetPortfolio(weights=weights)

