"""
PPO Strategy - RL Inference Engine

This module provides the PPOStrategy class that loads a trained PPO model
and generates target portfolios from live market data.

Critical: This class handles normalization drift by loading and applying
the VecNormalize statistics from training to ensure the agent sees data
in the same format it was trained on.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from execution.models import TargetPortfolio
from strategy.rl.fast_env import FastCryptoEnv


class PPOStrategy:
    """
    PPO-based strategy that uses a trained reinforcement learning agent
    to generate target portfolio allocations.
    
    This class handles the critical normalization step: the agent was trained
    on normalized data (via VecNormalize), so live data must be normalized
    using the same statistics before prediction.
    """
    
    def __init__(
        self,
        model_path: Path,
        vec_norm_path: Path,
        data_path: Path,
        asset_list: Optional[list] = None,
        provider: Optional[any] = None
    ):
        """
        Initialize the PPO strategy.
        
        Args:
            model_path: Path to trained PPO model (.zip file)
            vec_norm_path: Path to VecNormalize statistics (.pkl file)
            data_path: Path to training data (parquet file) - used for structure/fallback
            asset_list: List of asset tickers (default: auto-detect from data)
            provider: Optional ExecutionProvider for fetching live data (if None, uses static data_path)
        
        Raises:
            FileNotFoundError: If model or normalization files don't exist
        """
        self.data_path = data_path
        self.model_path = model_path
        self.vec_norm_path = vec_norm_path
        self.provider = provider  # For live data fetching
        
        # Verify files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vec_norm_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {vec_norm_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"📦 Loading PPO model from: {model_path}")
        
        # Load the trained PPO model
        self.model = PPO.load(str(model_path))
        
        print(f"✓ PPO model loaded")
        
        # Load normalization statistics (CRITICAL)
        print(f"📦 Loading normalization statistics from: {vec_norm_path}")
        
        # To load VecNormalize, we need a dummy environment that matches training structure
        # Create a temporary FastCryptoEnv to get the correct structure
        temp_env = FastCryptoEnv(data_path=data_path)
        
        # Get asset list from environment
        if asset_list is None:
            self.asset_list = temp_env.assets
        else:
            self.asset_list = asset_list
        
        self.n_assets = len(self.asset_list)
        
        # Create a dummy vectorized environment for VecNormalize
        # We need to pass a factory function, not the env itself
        def make_dummy_env():
            return temp_env
        
        dummy_vec_env = DummyVecEnv([make_dummy_env])
        
        # Load the normalization statistics
        self.norm_env = VecNormalize.load(str(vec_norm_path), dummy_vec_env)
        
        # CRITICAL: Freeze the normalization stats
        # We don't want live trading to update the moving averages
        self.norm_env.training = False
        self.norm_env.norm_reward = False
        
        print(f"✓ Normalization statistics loaded and frozen")
        print(f"  Assets: {self.asset_list}")
        print(f"  Observation space: {self.norm_env.observation_space.shape}")
    
    def _prepare_observation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare observation vector from live data.
        
        This method replicates the feature engineering from FastCryptoEnv:
        - Returns (pct_change)
        - Volatility (rolling std)
        - Normalized commit counts
        - Utility scores
        
        Args:
            df: DataFrame with market data (must have columns matching training data)
        
        Returns:
            Observation array with shape (1, n_features) ready for normalization
        """
        # Ensure data is sorted by timestamp and ticker
        df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        # Get the most recent timestamp
        latest_timestamp = df['timestamp'].max()
        latest_data = df[df['timestamp'] == latest_timestamp]
        
        # Initialize observation array
        # Features per asset: [return, volatility, commit_count_norm, utility_score]
        # Plus: portfolio_value_norm, cash_weight
        n_features_per_asset = 4
        n_obs_features = (self.n_assets * n_features_per_asset) + 2
        obs = np.zeros(n_obs_features, dtype=np.float32)
        
        idx = 0
        
        # For each asset, extract features (matching FastCryptoEnv structure)
        # Note: asset_list may have symbols without /USD, but data file uses base symbols
        for asset in self.asset_list:
            # Strip /USD if present for data lookup (data file uses base symbols like "BTC")
            base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
            asset_data = latest_data[latest_data['ticker'] == base_asset]
            
            if len(asset_data) > 0:
                row = asset_data.iloc[0]
                
                # Calculate returns (pct_change)
                # Get previous row for this asset to calculate return
                # Use base_asset for data lookup (data uses "BTC", not "BTC/USD")
                asset_history = df[df['ticker'] == base_asset].sort_values('timestamp')
                if len(asset_history) > 1:
                    current_price = asset_history.iloc[-1]['close']
                    previous_price = asset_history.iloc[-2]['close']
                    returns = (current_price / previous_price) - 1.0 if previous_price > 0 else 0.0
                else:
                    returns = 0.0
                
                # Calculate volatility (rolling std of returns)
                if len(asset_history) > 7:
                    asset_history['returns'] = asset_history['close'].pct_change()
                    volatility = asset_history['returns'].rolling(window=7, min_periods=1).std().iloc[-1]
                    volatility = volatility if not pd.isna(volatility) else 0.0
                else:
                    volatility = 0.0
                
                # Normalize commit_count (MinMax scaling 0-1)
                commit_count = row.get('commit_count_4_weeks', 0)
                # Use base_asset for data lookup (data uses "BTC", not "BTC/USD")
                asset_all_data = df[df['ticker'] == base_asset]
                commit_min = asset_all_data['commit_count_4_weeks'].min()
                commit_max = asset_all_data['commit_count_4_weeks'].max()
                if commit_max > commit_min:
                    commit_count_norm = (commit_count - commit_min) / (commit_max - commit_min)
                else:
                    commit_count_norm = 0.5
                
                # Utility score (already 0-1, but ensure valid)
                utility_score = row.get('utility_score', 0.5)
                utility_score = max(0.0, min(1.0, float(utility_score)))
                
                obs[idx] = returns
                obs[idx + 1] = volatility
                obs[idx + 2] = commit_count_norm
                obs[idx + 3] = utility_score
            else:
                # Asset not available, use zeros
                obs[idx:idx+4] = 0.0
            
            idx += 4
        
        # Add portfolio value (normalized to initial cash)
        # For live trading, we'll use 1.0 (current portfolio value / initial)
        # This should be updated from actual portfolio state in production
        obs[idx] = 1.0  # portfolio_value_norm
        
        # Add cash weight (will be set by agent, but initialize to 1.0 for safety)
        obs[idx + 1] = 1.0  # cash_weight
        
        # Reshape to (1, n_features) for batch prediction
        return obs.reshape(1, -1)
    
    def _fetch_live_data(self) -> pd.DataFrame:
        """
        Fetch live market data from Alpaca API.
        
        This is the "Live Data Bridge" that ensures the PPO agent sees current
        market prices instead of stale static data.
        
        Returns:
            pd.DataFrame: DataFrame with same structure as training data
                Columns: timestamp, ticker, open, high, low, close, volume, 
                         commit_count_4_weeks, utility_score
        
        Raises:
            RuntimeError: If provider is not set
            ValueError: If data fetching fails
        """
        if self.provider is None:
            # Fallback to static data if provider not available
            print("⚠ Warning: No provider set, using static data from parquet file")
            return pd.read_parquet(self.data_path)
        
        # Convert asset list to Alpaca symbols (add /USD if needed)
        alpaca_symbols = []
        for asset in self.asset_list:
            # Strip /USD if present, then add it back for Alpaca
            base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
            alpaca_symbols.append(f"{base_asset}/USD")
        
        print(f"📡 Fetching live market data for {len(alpaca_symbols)} assets...")
        
        # Fetch enough history for feature calculation
        # For 15Min bars: 500 hours = 2000 bars (~83 days)
        # This ensures we can calculate rolling volatility and returns correctly
        df = self.provider.fetch_crypto_history(
            symbols=alpaca_symbols,
            timeframe="15Min",  # v1.2: High-frequency 15-minute bars
            hours=500,  # Enough for rolling features
            include_fundamentals=False  # We'll use synthetic/neutral values
        )
        
        # Ensure we have the required columns
        required_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in live data: {missing_cols}")
        
        # Fill missing fundamental columns with neutral values (for feature engineering)
        if 'commit_count_4_weeks' not in df.columns:
            df['commit_count_4_weeks'] = 100.0  # Neutral placeholder
        if 'utility_score' not in df.columns:
            df['utility_score'] = 0.5  # Neutral sentiment
        
        print(f"✓ Live data fetched: {len(df)} bars across {df['ticker'].nunique()} assets")
        return df
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio using the trained PPO agent.
        
        This method now uses live market data from Alpaca API instead of
        static parquet files, ensuring the agent sees current prices.
        
        Args:
            portfolio_value: Current portfolio value (for normalization)
                            Default: 100000.0 (initial cash)
        
        Returns:
            TargetPortfolio with weights for each asset
        
        Raises:
            ValueError: If prediction fails or weights are invalid
        """
        # Fetch live data (or fallback to static if provider not set)
        df = self._fetch_live_data()
        
        # Prepare raw observation
        raw_obs = self._prepare_observation(df)
        
        # CRITICAL: Normalize observation using training statistics
        # This ensures the agent sees data in the same format it was trained on
        norm_obs = self.norm_env.normalize_obs(raw_obs)
        
        # Predict action (portfolio weights)
        # deterministic=True for consistent behavior in production
        action, _ = self.model.predict(norm_obs, deterministic=True)
        
        # Post-process action
        # Action is a vector of weights: [Asset1, Asset2, ..., AssetN, Cash]
        action = action.flatten()
        
        # Apply Halal constraints
        # 1. Ensure weights are positive (Long Only)
        action = np.clip(action, 0.0, 1.0)
        
        # 2. Normalize to sum to 1.0 (No Leverage)
        action_sum = np.sum(action)
        if action_sum > 1e-8:
            action = action / action_sum
        else:
            # All zeros or invalid, default to cash
            action = np.zeros(len(action))
            action[-1] = 1.0
        
        # Map weights to asset dictionary
        # CRITICAL: Convert asset symbols to Alpaca-compliant format (add /USD for crypto)
        # This ensures the Execution Engine recognizes them as crypto and uses correct API endpoints
        weights_dict = {}
        for i, asset in enumerate(self.asset_list):
            weight = float(action[i])
            if weight > 1e-6:  # Only include non-negligible weights
                # Convert to Alpaca crypto format: "BTC" -> "BTC/USD"
                # This allows the Execution Engine to properly identify crypto vs stocks
                if '/' not in asset and not asset.endswith('USD'):
                    # Add /USD suffix for crypto symbols
                    alpaca_symbol = f"{asset}/USD"
                else:
                    # Already in correct format or is "CASH"
                    alpaca_symbol = asset
                weights_dict[alpaca_symbol] = weight
        
        # Cash weight is the last element
        # Note: TargetPortfolio doesn't have a cash field, but we can
        # represent it as ensuring weights sum to <= 1.0
        # The remaining weight (1.0 - sum(weights)) is implicitly cash
        
        # Construct and return TargetPortfolio
        try:
            target = TargetPortfolio(weights=weights_dict)
            return target
        except Exception as e:
            # If validation fails, return empty portfolio (100% cash)
            print(f"⚠ Warning: TargetPortfolio validation failed: {e}")
            print(f"  Returning empty portfolio (100% cash)")
            return TargetPortfolio(weights={})

