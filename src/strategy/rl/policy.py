"""
Multimodal Policy Extractor - Two-Tower Architecture

Research Track Sprint 1: Semantic Grounding with Embeddings

This module implements a custom feature extractor for Stable-Baselines3 that
processes multimodal observations (market + text) using a two-tower architecture:

1. Market Tower: MLP processing price/volatility features -> 64 dim
2. Text Tower: MLP processing FinBERT embeddings -> 64 dim
3. Fusion: Concatenate (64, 64) -> 128 dim -> Policy

This architecture allows the agent to learn separate representations for
market and text modalities before combining them for decision-making.
"""

from typing import Dict

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultimodalExtractor(BaseFeaturesExtractor):
    """
    Multimodal feature extractor for market + text observations.
    
    Architecture:
        Market Tower: MLP(150) -> 128 -> 64
        Text Tower: MLP(768) -> 256 -> 64
        Fusion: Concat(64, 64) -> 128
    
    This two-tower design allows the agent to:
    1. Learn separate representations for market and text
    2. Fuse them at a higher abstraction level
    3. Make decisions based on combined understanding
    
    Example:
        extractor = MultimodalExtractor(
            observation_space=Dict({
                'market': Box(150,),
                'text': Box(768,)
            })
        )
        obs = {'market': market_features, 'text': text_embeddings}
        features = extractor(obs)  # Shape: (batch_size, 128)
    """
    
    def __init__(
        self,
        observation_space,
        market_dim: int = 150,
        text_dim: int = 768,
        market_hidden: int = 128,
        text_hidden: int = 256,
        output_dim: int = 64,
        fusion_dim: int = 128
    ):
        """
        Initialize the multimodal extractor.
        
        Args:
            observation_space: Gymnasium Dict space with 'market' and 'text' keys
            market_dim: Dimension of market features (default: 150)
            text_dim: Dimension of text embeddings (default: 768)
            market_hidden: Hidden dimension for market tower (default: 128)
            text_hidden: Hidden dimension for text tower (default: 256)
            output_dim: Output dimension for each tower (default: 64)
            fusion_dim: Final fusion dimension (default: 128)
        """
        # Calculate total feature dimension for BaseFeaturesExtractor
        total_features_dim = fusion_dim
        
        super().__init__(observation_space, features_dim=total_features_dim)
        
        # Market Tower: Process price/volatility features
        self.market_tower = nn.Sequential(
            nn.Linear(market_dim, market_hidden),
            nn.ReLU(),
            nn.Linear(market_hidden, output_dim),
            nn.ReLU()
        )
        
        # Text Tower: Process FinBERT embeddings
        self.text_tower = nn.Sequential(
            nn.Linear(text_dim, text_hidden),
            nn.ReLU(),
            nn.Linear(text_hidden, output_dim),
            nn.ReLU()
        )
        
        # Fusion Layer: Combine market and text representations
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, fusion_dim),  # 64 + 64 = 128
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal extractor.
        
        Args:
            observations: Dict with keys:
                - 'market': Market features tensor (batch_size, market_dim)
                - 'text': Text embeddings tensor (batch_size, text_dim)
        
        Returns:
            torch.Tensor: Fused features (batch_size, fusion_dim)
        """
        # Extract market and text features
        market_features = observations['market']
        text_features = observations['text']
        
        # Process through respective towers
        market_encoded = self.market_tower(market_features)  # (batch, 64)
        text_encoded = self.text_tower(text_features)  # (batch, 64)
        
        # Concatenate and fuse
        combined = torch.cat([market_encoded, text_encoded], dim=1)  # (batch, 128)
        fused = self.fusion(combined)  # (batch, fusion_dim)
        
        return fused

