"""
Vector Environment - Multimodal RL Environment with Text Embeddings

Research Track Sprint 1: Semantic Grounding with Embeddings

This module extends FastCryptoEnv to support multimodal observations:
- Market features (price, volatility, etc.) - 150 dimensions
- Text embeddings (FinBERT) - 768 dimensions

The environment uses a Dict observation space to separate market and text modalities,
enabling the multimodal policy to process them independently before fusion.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces

from strategy.rl.fast_env import FastCryptoEnv


class CryptoVectorEnv(FastCryptoEnv):
    """
    Multimodal crypto trading environment with text embeddings.
    
    Extends FastCryptoEnv to include 768-dimensional FinBERT embeddings
    alongside market features. Uses Dict observation space for clean
    separation of modalities.
    
    Observation Space:
        Dict({
            'market': Box(shape=(n_market_features,)),  # Price, volatility, etc.
            'text': Box(shape=(768,))  # FinBERT embeddings
        })
    
    This structure allows the multimodal policy to process market and text
    features independently before fusion.
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        data_config: Optional[Dict] = None,
        embeddings_path: Optional[Path] = None,
        lookback_window: int = 30,
        initial_cash: float = 100000.0,
        fee_rate: float = 0.001,
        benchmark_returns: Optional[np.ndarray] = None
    ):
        """
        Initialize the multimodal vector environment.
        
        Args:
            data_path: Path to training data (used if data_config is None)
            data_config: Preprocessed data configuration (same as FastCryptoEnv)
            embeddings_path: Path to news_embeddings.npy (shape: N_timesteps, N_assets, 768)
            lookback_window: Number of days for Sortino calculation
            initial_cash: Starting portfolio value
            fee_rate: Transaction fee rate (default: 0.001 = 0.1%)
            benchmark_returns: Optional benchmark returns for DIR calculation
        """
        # Initialize parent environment (FastCryptoEnv)
        super().__init__(
            data_path=data_path,
            data_config=data_config,
            lookback_window=lookback_window,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            benchmark_returns=benchmark_returns
        )
        
        # Load embeddings if provided
        self.has_embeddings = False
        self.embeddings_array = None
        
        if embeddings_path is not None and embeddings_path.exists():
            print(f"📚 Loading text embeddings from: {embeddings_path}")
            self.embeddings_array = np.load(str(embeddings_path), mmap_mode='r')
            
            # Verify shape compatibility
            expected_shape = (self.n_steps, self.n_assets, 768)
            if self.embeddings_array.shape != expected_shape:
                raise ValueError(
                    f"Embeddings shape mismatch: "
                    f"expected {expected_shape}, got {self.embeddings_array.shape}"
                )
            
            self.has_embeddings = True
            print(f"  ✓ Loaded embeddings: {self.embeddings_array.shape}")
        elif data_config is not None and 'embeddings_path' in data_config:
            # Load from data_config
            embeddings_path = Path(data_config['embeddings_path'])
            if embeddings_path.exists():
                print(f"📚 Loading text embeddings from data_config...")
                self.embeddings_array = np.load(str(embeddings_path), mmap_mode='r')
                self.has_embeddings = True
                print(f"  ✓ Loaded embeddings: {self.embeddings_array.shape}")
        
        if not self.has_embeddings:
            print("  ⚠ No embeddings provided, using zero vectors (text modality disabled)")
            # Create zero embeddings array for backward compatibility
            self.embeddings_array = np.zeros(
                (self.n_steps, self.n_assets, 768),
                dtype=np.float32
            )
        
        # Override observation space to use Dict space
        # Get market feature dimension from parent
        market_obs_dim = self.observation_space.shape[0] if hasattr(self.observation_space, 'shape') else 150
        
        # Create Dict observation space
        self.observation_space = spaces.Dict({
            'market': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(market_obs_dim,),
                dtype=np.float32
            ),
            'text': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(768,),
                dtype=np.float32
            )
        })
        
        print(f"✓ Multimodal observation space: market={market_obs_dim}, text=768")
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get multimodal observation (market + text).
        
        Returns:
            Dict with keys:
                - 'market': Market features (price, volatility, etc.)
                - 'text': Text embeddings (FinBERT, 768-dim)
        """
        # Get market observation from parent
        market_obs = super()._get_observation()
        
        # Get text embedding for current timestep
        if self.has_embeddings and self.current_step < self.n_steps:
            # Aggregate embeddings across assets (weighted by current portfolio weights)
            # For simplicity, use equal weighting or current portfolio weights
            if hasattr(self, 'current_weights') and self.current_weights is not None:
                # Weighted average by portfolio allocation
                weights = self.current_weights[:self.n_assets]  # Exclude cash
                weights = weights / (weights.sum() + 1e-8)  # Normalize
                text_embedding = np.dot(weights, self.embeddings_array[self.current_step, :, :])
            else:
                # Equal weighting (mean pooling across assets)
                text_embedding = self.embeddings_array[self.current_step, :, :].mean(axis=0)
        else:
            # Zero vector if no embeddings
            text_embedding = np.zeros(768, dtype=np.float32)
        
        return {
            'market': market_obs,
            'text': text_embedding
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment and return initial multimodal observation."""
        # Reset parent environment
        obs, info = super().reset(seed=seed, options=options)
        
        # Convert to Dict observation
        multimodal_obs = self._get_observation()
        
        return multimodal_obs, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio weights (same as FastCryptoEnv)
        
        Returns:
            Tuple (observation, reward, terminated, truncated, info)
            where observation is a Dict with 'market' and 'text' keys
        """
        # Execute step in parent environment
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Convert to Dict observation
        multimodal_obs = self._get_observation()
        
        return multimodal_obs, reward, terminated, truncated, info

