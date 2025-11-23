"""
Crypto Deep RL Environment - Custom Gymnasium Environment

This module implements a custom trading environment for training a PPO agent
on multi-modal crypto data (Price + Fundamentals + Sentiment).

The agent learns to allocate portfolio weights across assets to maximize
the Sortino Ratio (risk-adjusted return with focus on downside protection).
"""

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class CryptoDeepEnv(gym.Env):
    """
    Custom Gymnasium environment for crypto portfolio trading.
    
    The agent observes:
    - Technical signals (returns, volatility)
    - Fundamental signals (developer activity)
    - Sentiment signals (utility score)
    
    The agent outputs portfolio weights (action space).
    The reward is the Sortino Ratio (risk-adjusted return).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        lookback_window: int = 30,
        initial_cash: float = 100000.0
    ):
        """
        Initialize the crypto trading environment.
        
        Args:
            data_path: Path to crypto_training_data.parquet (default: auto-detect)
            lookback_window: Number of days for Sortino calculation (default: 30)
            initial_cash: Starting portfolio value (default: 100000)
        """
        super().__init__()
        
        # Set data path
        if data_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
        
        self.data_path = data_path
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        
        # Load and preprocess data
        print("📊 Loading crypto training data...")
        self._load_and_preprocess()
        
        # Get unique assets
        self.assets = sorted(self.df['ticker'].unique().tolist())
        self.n_assets = len(self.assets)
        
        print(f"✓ Loaded {len(self.df)} rows for {self.n_assets} assets: {self.assets}")
        
        # Define action space: Portfolio weights (assets + cash)
        # Shape: (n_assets + 1,) where last element is cash weight
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets + 1,),
            dtype=np.float32
        )
        
        # Define observation space
        # Features per asset: [return, volatility, commit_count_norm, utility_score]
        # Plus: current_portfolio_value, cash_weight
        n_features_per_asset = 4
        n_obs_features = (self.n_assets * n_features_per_asset) + 2  # +2 for portfolio_value and cash
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs_features,),
            dtype=np.float32
        )
        
        # Initialize episode state
        self.current_step = 0
        self.returns_history = deque(maxlen=lookback_window)
        self.portfolio_value = initial_cash
        self.current_weights = np.zeros(self.n_assets + 1)
        self.current_weights[-1] = 1.0  # Start with 100% cash
        
        # Episode tracking
        self.episode_returns = []
        self.episode_length = 0
    
    def _load_and_preprocess(self):
        """Load and preprocess the crypto dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                "Please run scripts/09_generate_sentiment.py first."
            )
        
        df = pd.read_parquet(self.data_path)
        
        # Sort by timestamp and ticker
        df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        # Calculate technical indicators per asset
        processed_data = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Calculate returns (pct_change)
            ticker_data['returns'] = ticker_data['close'].pct_change()
            
            # Calculate volatility (rolling std of returns)
            ticker_data['volatility'] = ticker_data['returns'].rolling(window=7, min_periods=1).std()
            
            # Normalize commit_count_4_weeks (MinMax scaling 0-1)
            commit_min = ticker_data['commit_count_4_weeks'].min()
            commit_max = ticker_data['commit_count_4_weeks'].max()
            if commit_max > commit_min:
                ticker_data['commit_count_norm'] = (
                    (ticker_data['commit_count_4_weeks'] - commit_min) / (commit_max - commit_min)
                )
            else:
                ticker_data['commit_count_norm'] = 0.5
            
            # utility_score is already 0-1, but ensure it's not NaN
            ticker_data['utility_score'] = ticker_data['utility_score'].fillna(0.5)
            
            # Fill NaN returns and volatility with 0
            ticker_data['returns'] = ticker_data['returns'].fillna(0.0)
            ticker_data['volatility'] = ticker_data['volatility'].fillna(0.0)
            
            processed_data.append(ticker_data)
        
        # Combine all assets
        self.df = pd.concat(processed_data, ignore_index=True)
        self.df = self.df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        # Get unique timestamps (episode steps)
        self.timestamps = sorted(self.df['timestamp'].unique())
        self.n_steps = len(self.timestamps)
        
        print(f"✓ Preprocessed data: {self.n_steps} time steps")
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector for the current step.
        
        Returns:
            Observation vector: [Asset1_Return, Asset1_Vol, Asset1_Commit, Asset1_Utility, 
                               Asset2_..., ..., Portfolio_Value, Cash_Weight]
        """
        if self.current_step >= len(self.timestamps):
            # End of episode, return zeros
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        current_time = self.timestamps[self.current_step]
        current_data = self.df[self.df['timestamp'] == current_time]
        
        obs = []
        
        # For each asset, extract features
        for asset in self.assets:
            asset_data = current_data[current_data['ticker'] == asset]
            
            if len(asset_data) > 0:
                row = asset_data.iloc[0]
                obs.extend([
                    row['returns'],
                    row['volatility'],
                    row['commit_count_norm'],
                    row['utility_score']
                ])
            else:
                # Asset not available at this timestamp, use zeros
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # Add portfolio value (normalized)
        obs.append(self.portfolio_value / self.initial_cash)
        
        # Add current cash weight
        obs.append(self.current_weights[-1])
        
        return np.array(obs, dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Info dictionary
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        self.returns_history.clear()
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.n_assets + 1)
        self.current_weights[-1] = 1.0  # Start with 100% cash
        self.episode_returns = []
        self.episode_length = 0
        
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'step': self.current_step
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio weights (must sum to 1.0)
        
        Returns:
            observation: Next observation
            reward: Reward (Sortino Ratio or daily return)
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Info dictionary
        """
        # Normalize action to ensure weights sum to 1.0
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum > 0:
            weights = action / action_sum
        else:
            # All zeros, default to cash
            weights = np.zeros(self.n_assets + 1)
            weights[-1] = 1.0
        
        self.current_weights = weights
        
        # Check if we've reached the end
        if self.current_step >= len(self.timestamps) - 1:
            # End of episode
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'portfolio_value': self.portfolio_value,
                'step': self.current_step,
                'episode_length': self.episode_length,
                'final_return': (self.portfolio_value / self.initial_cash) - 1.0
            }
            return observation, reward, terminated, truncated, info
        
        # Get current and next timestamps
        current_time = self.timestamps[self.current_step]
        next_time = self.timestamps[self.current_step + 1]
        
        # Get asset returns for the next period
        current_data = self.df[self.df['timestamp'] == current_time]
        next_data = self.df[self.df['timestamp'] == next_time]
        
        # Calculate portfolio return
        portfolio_return = 0.0
        
        for i, asset in enumerate(self.assets):
            asset_weight = weights[i]
            if asset_weight > 0:
                asset_current = current_data[current_data['ticker'] == asset]
                asset_next = next_data[next_data['ticker'] == asset]
                
                if len(asset_current) > 0 and len(asset_next) > 0:
                    current_price = asset_current.iloc[0]['close']
                    next_price = asset_next.iloc[0]['close']
                    asset_return = (next_price / current_price) - 1.0
                    portfolio_return += asset_weight * asset_return
        
        # Update portfolio value
        self.portfolio_value *= (1.0 + portfolio_return)
        
        # Track returns history
        self.returns_history.append(portfolio_return)
        self.episode_returns.append(portfolio_return)
        self.episode_length += 1
        
        # Calculate reward
        if len(self.returns_history) < self.lookback_window:
            # Not enough history, use daily return as reward
            reward = portfolio_return
        else:
            # Calculate Sortino Ratio
            returns_array = np.array(list(self.returns_history))
            mean_return = np.mean(returns_array)
            
            # Downside deviation: std of negative returns only
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) > 0 and np.std(downside_returns) > 1e-8:
                downside_dev = np.std(downside_returns)
                sortino_ratio = mean_return / downside_dev
                reward = sortino_ratio
            else:
                # No downside or zero deviation, use mean return
                reward = mean_return
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= len(self.timestamps) - 1
        truncated = False
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'step': self.current_step,
            'episode_length': self.episode_length,
            'sortino_ratio': reward if len(self.returns_history) >= self.lookback_window else None
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional, for debugging)."""
        if self.current_step < len(self.timestamps):
            print(f"Step {self.current_step}: Portfolio Value = ${self.portfolio_value:.2f}")

