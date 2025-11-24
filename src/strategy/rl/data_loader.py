"""
Data Preprocessing for FastCryptoEnv - Load Once, Use Many

This module provides a function to preprocess the crypto training data
ONCE in the main process, then pass the clean numpy arrays to multiple
environment instances. This eliminates redundant file I/O and processing
when using SubprocVecEnv with parallel environments.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def prepare_training_data(data_path: Path) -> Dict:
    """
    Load and preprocess crypto training data into optimized numpy arrays.
    
    This function should be called ONCE in the main process before spawning
    parallel environments. The returned dictionary can be passed to FastCryptoEnv
    instances to avoid redundant loading and processing.
    
    Args:
        data_path: Path to crypto_training_data.parquet
    
    Returns:
        Dictionary with preprocessed arrays:
            - 'returns_array': [n_steps, n_assets] float32
            - 'volatility_array': [n_steps, n_assets] float32
            - 'commit_array': [n_steps, n_assets] float32
            - 'utility_array': [n_steps, n_assets] float32
            - 'price_array': [n_steps, n_assets] float32
            - 'asset_list': List of asset symbols
            - 'timestamps': List of timestamps
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Please run scripts/09_generate_sentiment.py first."
        )
    
    print("📊 Loading and optimizing crypto training data (Main Process)...")
    
    # Load parquet file
    df = pd.read_parquet(data_path)
    df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
    
    # Get unique assets and timestamps
    asset_list = sorted(df['ticker'].unique().tolist())
    timestamps = sorted(df['timestamp'].unique())
    n_steps = len(timestamps)
    n_assets = len(asset_list)
    
    print(f"  Found {n_steps} timestamps, {n_assets} assets")
    
    # Pre-allocate numpy arrays for maximum speed
    # Shape: [n_steps, n_assets]
    returns_array = np.zeros((n_steps, n_assets), dtype=np.float32)
    volatility_array = np.zeros((n_steps, n_assets), dtype=np.float32)
    commit_array = np.zeros((n_steps, n_assets), dtype=np.float32)
    utility_array = np.zeros((n_steps, n_assets), dtype=np.float32)
    price_array = np.zeros((n_steps, n_assets), dtype=np.float32)
    
    # Process each asset
    for asset_idx, asset in enumerate(asset_list):
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
        for ts_idx, timestamp in enumerate(timestamps):
            ts_data = asset_data[asset_data['timestamp'] == timestamp]
            if len(ts_data) > 0:
                row = ts_data.iloc[0]
                returns_array[ts_idx, asset_idx] = float(row['returns'])
                volatility_array[ts_idx, asset_idx] = float(row['volatility'])
                commit_array[ts_idx, asset_idx] = float(row['commit_count_norm'])
                utility_array[ts_idx, asset_idx] = float(row['utility_score'])
                price_array[ts_idx, asset_idx] = float(row['close'])
    
    # Calculate total memory usage
    total_bytes = (
        returns_array.nbytes +
        volatility_array.nbytes +
        commit_array.nbytes +
        utility_array.nbytes +
        price_array.nbytes
    )
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"✅ Data optimized. Shape: [{n_steps}, {n_assets}]. Size: {total_mb:.2f} MB")
    
    return {
        'returns_array': returns_array,
        'volatility_array': volatility_array,
        'commit_array': commit_array,
        'utility_array': utility_array,
        'price_array': price_array,
        'asset_list': asset_list,
        'timestamps': timestamps
    }

