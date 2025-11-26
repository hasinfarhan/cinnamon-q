"""
Buy and Hold Strategy - The Passive Benchmark

This strategy implements equal-weight indexing: every asset in the universe
gets the same allocation. This is the "lazy investor" baseline.

Why: If your AI can't beat "doing nothing," it's not adding value.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from execution.models import TargetPortfolio


class BuyAndHoldStrategy:
    """
    Equal-weight buy and hold strategy.
    
    This strategy assigns equal weights to all assets in the universe,
    fully invested (no cash). This serves as a passive baseline.
    
    Attributes:
        asset_list: List of asset symbols to hold
        data_path: Path to data file (for universe loading, not used for signals)
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        universe_csv: Optional[Path] = None
    ):
        """
        Initialize the Buy and Hold strategy.
        
        Args:
            data_path: Path to data file (optional, for compatibility)
            universe_csv: Path to crypto_universe.csv (optional, auto-detected if None)
        """
        self.data_path = data_path
        
        # Load asset universe
        if universe_csv is None:
            # Auto-detect path
            if data_path:
                project_root = data_path.parent.parent.parent
            else:
                project_root = Path(__file__).parent.parent.parent.parent
            universe_csv = project_root / "data" / "raw" / "crypto_universe.csv"
        
        if not universe_csv.exists():
            # Fallback: Load from data file
            if data_path and Path(data_path).exists():
                df = pd.read_parquet(data_path)
                self.asset_list = sorted(df['ticker'].unique().tolist())
            else:
                raise FileNotFoundError(
                    f"Universe CSV not found: {universe_csv}\n"
                    "Please ensure data/raw/crypto_universe.csv exists."
                )
        else:
            # Load from CSV
            df = pd.read_csv(universe_csv)
            if 'alpaca_symbol' in df.columns:
                self.asset_list = sorted(df['alpaca_symbol'].dropna().unique().tolist())
            elif 'ticker' in df.columns:
                self.asset_list = sorted(df['ticker'].dropna().unique().tolist())
            else:
                raise ValueError(f"Universe CSV missing required column: {universe_csv}")
        
        # Convert to base symbols (remove /USD if present)
        self.asset_list = [asset.replace('/USD', '') if '/USD' in asset else asset 
                          for asset in self.asset_list]
        self.asset_list = sorted(self.asset_list)
        
        print(f"✓ Buy and Hold Strategy initialized")
        print(f"  Assets: {len(self.asset_list)}")
        print(f"  Allocation: Equal weight ({1.0/len(self.asset_list)*100:.2f}% per asset)")
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio with equal weights for all assets.
        
        Args:
            portfolio_value: Current portfolio value (unused, for compatibility)
        
        Returns:
            TargetPortfolio with equal weights for all assets, 0% cash
        """
        n_assets = len(self.asset_list)
        if n_assets == 0:
            # No assets, 100% cash
            return TargetPortfolio(weights={"CASH": 1.0})
        
        # Equal weight per asset
        weight_per_asset = 1.0 / n_assets
        
        # Create weights dictionary
        weights = {asset: weight_per_asset for asset in self.asset_list}
        weights["CASH"] = 0.0  # Fully invested
        
        return TargetPortfolio(weights=weights)

