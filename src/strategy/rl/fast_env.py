"""
Fast Crypto RL Environment - Optimized Numpy Version

This is a high-performance version of CryptoDeepEnv that uses Numpy arrays
instead of Pandas DataFrames for blazingly fast stepping. Designed for
parallel training with SubprocVecEnv.
"""

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class FastCryptoEnv(gym.Env):
    """
    Highly optimized crypto trading environment using Numpy arrays.
    
    This version pre-processes all data into float32 arrays for maximum
    performance during training. Designed for parallel execution.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        data_config: Optional[Dict] = None,
        lookback_window: int = 30,
        initial_cash: float = 100000.0
    ):
        """
        Initialize the fast crypto trading environment.
        
        Args:
            data_path: Path to crypto_training_data.parquet (used if data_config is None)
            data_config: Preprocessed data configuration with keys:
                - 'returns_path': Path to returns_array.npy (memory-mapped)
                - 'volatility_path': Path to volatility_array.npy (memory-mapped)
                - 'commit_path': Path to commit_array.npy (memory-mapped)
                - 'utility_path': Path to utility_array.npy (memory-mapped)
                - 'price_path': Path to price_array.npy (memory-mapped)
                - 'asset_list': List of asset symbols
                - 'timestamps': List of timestamps
                - 'n_steps': Number of timesteps
                - 'n_assets': Number of assets
            lookback_window: Number of days for Sortino calculation
            initial_cash: Starting portfolio value
        """
        super().__init__()
        
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        
        # Use memory-mapped files if provided, otherwise load from parquet
        if data_config is not None:
            # Lightweight initialization: load from memory-mapped files (zero-copy)
            # mmap_mode='r' means read-only, shared memory access
            self.returns_array = np.load(data_config['returns_path'], mmap_mode='r')
            self.volatility_array = np.load(data_config['volatility_path'], mmap_mode='r')
            self.commit_array = np.load(data_config['commit_path'], mmap_mode='r')
            self.utility_array = np.load(data_config['utility_path'], mmap_mode='r')
            self.price_array = np.load(data_config['price_path'], mmap_mode='r')
            self.asset_list = data_config['asset_list']
            self.timestamps = data_config['timestamps']
            self.n_steps = data_config['n_steps']
            self.has_ta_features = data_config.get('has_ta_features', False)
            
            # Load TA arrays if present
            if self.has_ta_features:
                self.rsi_array = np.load(data_config['rsi_path'], mmap_mode='r')
                self.macd_array = np.load(data_config['macd_path'], mmap_mode='r')
                self.bbw_array = np.load(data_config['bbw_path'], mmap_mode='r')
                self.atr_array = np.load(data_config['atr_path'], mmap_mode='r')
                self.sma_ratio_array = np.load(data_config['sma_ratio_path'], mmap_mode='r')
            else:
                # Create zero arrays for backward compatibility
                self.rsi_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.macd_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.bbw_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.atr_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.sma_ratio_array = np.ones((self.n_steps, len(self.asset_list)), dtype=np.float32)
        else:
            # Legacy mode: load from file (for backward compatibility)
            if data_path is None:
                project_root = Path(__file__).parent.parent.parent.parent
                data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
            
            self.data_path = data_path
            print("📊 Loading and optimizing crypto training data...")
            self._load_and_optimize()
        
        # Get unique assets
        self.assets = sorted(self.asset_list)
        self.n_assets = len(self.assets)
        
        print(f"✓ Optimized {self.n_steps} steps for {self.n_assets} assets: {self.assets}")
        
        # Define action space: Portfolio weights (assets + cash)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets + 1,),
            dtype=np.float32
        )
        
        # Define observation space DYNAMICALLY
        # Base features per asset: [return, volatility, commit_count_norm, utility_score] = 4
        # TA features per asset (if present): [RSI, MACD, BBW, ATR, SMA_RATIO] = 5
        # Plus: portfolio_value_norm, cash_weight = 2
        
        base_features_per_asset = 4
        ta_features_per_asset = 5 if (hasattr(self, 'has_ta_features') and self.has_ta_features) else 0
        n_features_per_asset = base_features_per_asset + ta_features_per_asset
        n_obs_features = (self.n_assets * n_features_per_asset) + 2
        
        self.n_features_per_asset = n_features_per_asset  # Store for observation construction
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs_features,),
            dtype=np.float32
        )
        
        if ta_features_per_asset > 0:
            print(f"  ✓ Observation space: {n_obs_features} features ({base_features_per_asset} base + {ta_features_per_asset} TA per asset)")
        else:
            print(f"  ✓ Observation space: {n_obs_features} features ({base_features_per_asset} base per asset, no TA)")
        
        # Initialize episode state
        self.current_step = 0
        self.returns_history = deque(maxlen=lookback_window)
        self.portfolio_value = initial_cash
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # Start with 100% cash
    
    def _load_and_optimize(self):
        """
        Load data and convert to optimized numpy arrays.
        
        This pre-processes everything into float32 arrays for maximum speed.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                "Please run scripts/09_generate_sentiment.py first."
            )
        
        df = pd.read_parquet(self.data_path)
        df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        # Get unique assets and timestamps
        self.asset_list = sorted(df['ticker'].unique().tolist())
        self.timestamps = sorted(df['timestamp'].unique())
        self.n_steps = len(self.timestamps)
        n_assets = len(self.asset_list)
        
        # Pre-allocate numpy arrays for maximum speed
        # Shape: [n_steps, n_assets, n_features]
        # Features: [returns, volatility, commit_count_norm, utility_score]
        n_features = 4
        
        # Data arrays
        self.returns_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        self.volatility_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        self.commit_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        self.utility_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        
        # Price arrays for portfolio return calculation
        self.price_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        
        # Process each asset
        for asset_idx, asset in enumerate(self.asset_list):
            asset_data = df[df['ticker'] == asset].sort_values('timestamp').reset_index(drop=True)
            
            # Calculate returns
            asset_data['returns'] = asset_data['close'].pct_change().fillna(0.0)
            
            # Calculate volatility (rolling std)
            asset_data['volatility'] = asset_data['returns'].rolling(window=7, min_periods=1).std().fillna(0.0)
            
            # Normalize commit_count
            commit_min = asset_data['commit_count_4_weeks'].min()
            commit_max = asset_data['commit_count_4_weeks'].max()
            if commit_max > commit_min:
                asset_data['commit_count_norm'] = (
                    (asset_data['commit_count_4_weeks'] - commit_min) / (commit_max - commit_min)
                )
            else:
                asset_data['commit_count_norm'] = 0.5
            
            # Ensure utility_score is valid
            asset_data['utility_score'] = asset_data['utility_score'].fillna(0.5).clip(0.0, 1.0)
            
            # Map to timestamps
            for ts_idx, timestamp in enumerate(self.timestamps):
                ts_data = asset_data[asset_data['timestamp'] == timestamp]
                if len(ts_data) > 0:
                    row = ts_data.iloc[0]
                    self.returns_array[ts_idx, asset_idx] = float(row['returns'])
                    self.volatility_array[ts_idx, asset_idx] = float(row['volatility'])
                    self.commit_array[ts_idx, asset_idx] = float(row['commit_count_norm'])
                    self.utility_array[ts_idx, asset_idx] = float(row['utility_score'])
                    self.price_array[ts_idx, asset_idx] = float(row['close'])
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector using fast numpy operations.
        
        Phase 5.1: Now includes TA features if available.
        Feature order (per asset): [returns, volatility, commit_count, utility, RSI, MACD, BBW, ATR, SMA_RATIO]
        
        Returns:
            Observation vector as float32 numpy array
        """
        if self.current_step >= self.n_steps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        idx = 0
        
        # For each asset, extract features (vectorized)
        has_ta = hasattr(self, 'has_ta_features') and self.has_ta_features
        
        for asset_idx in range(self.n_assets):
            # Base features (always present)
            obs[idx] = self.returns_array[self.current_step, asset_idx]
            obs[idx + 1] = self.volatility_array[self.current_step, asset_idx]
            obs[idx + 2] = self.commit_array[self.current_step, asset_idx]
            obs[idx + 3] = self.utility_array[self.current_step, asset_idx]
            idx += 4
            
            # TA features (if present)
            if has_ta:
                obs[idx] = self.rsi_array[self.current_step, asset_idx]
                obs[idx + 1] = self.macd_array[self.current_step, asset_idx]
                obs[idx + 2] = self.bbw_array[self.current_step, asset_idx]
                obs[idx + 3] = self.atr_array[self.current_step, asset_idx]
                obs[idx + 4] = self.sma_ratio_array[self.current_step, asset_idx]
                idx += 5
        
        # Add portfolio value (normalized)
        obs[idx] = self.portfolio_value / self.initial_cash
        
        # Add current cash weight
        obs[idx + 1] = self.current_weights[-1]
        
        return obs
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random start to prevent overfitting (but ensure we have enough history)
        if self.n_steps > 2000:
            self.current_step = np.random.randint(
                self.lookback_window,
                self.n_steps - 1000
            )
        else:
            self.current_step = self.lookback_window
        
        # Reset episode state
        self.returns_history.clear()
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        
        observation = self._get_observation()
        info = {
            'portfolio_value': float(self.portfolio_value),
            'step': int(self.current_step)
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step using optimized numpy operations.
        
        Args:
            action: Portfolio weights (shape: (n_assets + 1,))
        
        Returns:
            Standard gymnasium 5-tuple
        """
        # Ensure action is a numpy array and has correct shape
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Flatten action if it's multi-dimensional
        action = action.flatten()
        
        # Ensure action has correct length
        expected_len = self.n_assets + 1
        if len(action) != expected_len:
            raise ValueError(
                f"Action shape mismatch: expected {expected_len} (n_assets={self.n_assets} + 1), "
                f"got {len(action)}"
            )
        
        # Normalize action (fast numpy)
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum > 1e-8:
            weights = (action / action_sum).astype(np.float32)
        else:
            weights = np.zeros(self.n_assets + 1, dtype=np.float32)
            weights[-1] = 1.0
        
        self.current_weights = weights
        
        # Check termination
        if self.current_step >= self.n_steps - 1:
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'portfolio_value': float(self.portfolio_value),
                'step': int(self.current_step),
                'final_return': float((self.portfolio_value / self.initial_cash) - 1.0)
            }
            return observation, reward, terminated, truncated, info
        
        # Calculate portfolio return (vectorized numpy)
        if self.current_step < self.n_steps - 1:
            current_prices = self.price_array[self.current_step, :]
            next_prices = self.price_array[self.current_step + 1, :]
            
            # Avoid division by zero
            mask = current_prices > 1e-8
            asset_returns = np.zeros(self.n_assets, dtype=np.float32)
            asset_returns[mask] = (next_prices[mask] / current_prices[mask]) - 1.0
            
            # Portfolio return = dot product of weights and returns
            portfolio_return = np.dot(weights[:-1], asset_returns)  # Exclude cash weight
        else:
            portfolio_return = 0.0
        
        # Update portfolio value
        self.portfolio_value *= (1.0 + portfolio_return)
        
        # Track returns history
        self.returns_history.append(portfolio_return)
        
        # Calculate reward
        if len(self.returns_history) < self.lookback_window:
            reward = float(portfolio_return)
        else:
            # Calculate Sortino Ratio
            returns_array = np.array(list(self.returns_history), dtype=np.float32)
            mean_return = np.mean(returns_array)
            
            # Downside deviation
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) > 0 and np.std(downside_returns) > 1e-8:
                downside_dev = np.std(downside_returns)
                sortino_ratio = mean_return / downside_dev
                reward = float(sortino_ratio)
            else:
                reward = float(mean_return)
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        info = {
            'portfolio_value': float(self.portfolio_value),
            'portfolio_return': float(portfolio_return),  # CRITICAL: Actual percentage return for compounding
            'step': int(self.current_step),
            'sortino_ratio': reward if len(self.returns_history) >= self.lookback_window else None,
            'reward_type': 'sortino_ratio' if len(self.returns_history) >= self.lookback_window else 'portfolio_return'
        }
        
        return observation, reward, terminated, truncated, info

