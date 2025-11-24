"""
Data Preprocessing for FastCryptoEnv - Load Once, Use Many (Memory-Mapped)

This module provides a function to preprocess the crypto training data
ONCE in the main process, then save it to disk for memory-mapped access
by multiple environment instances. This eliminates redundant file I/O,
processing, and pickling overhead when using SubprocVecEnv.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def prepare_training_data(data_path: Path, temp_dir: Path) -> Dict:
    """
    Load and preprocess crypto training data, save to disk for memory-mapped access.
    
    This function should be called ONCE in the main process before spawning
    parallel environments. It saves the preprocessed arrays to disk, which
    subprocesses can then access via memory-mapped files (zero-copy, instant).
    
    Args:
        data_path: Path to crypto_training_data.parquet
        temp_dir: Temporary directory to save .npy files
    
    Returns:
        Dictionary with file paths and metadata:
            - 'returns_path': Path to returns_array.npy
            - 'volatility_path': Path to volatility_array.npy
            - 'commit_path': Path to commit_array.npy
            - 'utility_path': Path to utility_array.npy
            - 'price_path': Path to price_array.npy
            - 'asset_list': List of asset symbols
            - 'timestamps': List of timestamps
            - 'n_steps': Number of timesteps
            - 'n_assets': Number of assets
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
    
    # Save arrays to disk for memory-mapped access
    # This avoids pickling overhead when passing to subprocesses
    print("  Saving arrays to disk for memory-mapped access...")
    
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    returns_path = temp_dir / "returns_array.npy"
    volatility_path = temp_dir / "volatility_array.npy"
    commit_path = temp_dir / "commit_array.npy"
    utility_path = temp_dir / "utility_array.npy"
    price_path = temp_dir / "price_array.npy"
    
    np.save(str(returns_path), returns_array)
    np.save(str(volatility_path), volatility_array)
    np.save(str(commit_path), commit_array)
    np.save(str(utility_path), utility_array)
    np.save(str(price_path), price_array)
    
    # Calculate total memory usage
    total_bytes = (
        returns_array.nbytes +
        volatility_array.nbytes +
        commit_array.nbytes +
        utility_array.nbytes +
        price_array.nbytes
    )
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"✅ Data optimized and saved. Shape: [{n_steps}, {n_assets}]. Size: {total_mb:.2f} MB")
    print(f"   Files saved to: {temp_dir}")
    
    return {
        'returns_path': str(returns_path),
        'volatility_path': str(volatility_path),
        'commit_path': str(commit_path),
        'utility_path': str(utility_path),
        'price_path': str(price_path),
        'asset_list': asset_list,
        'timestamps': timestamps,
        'n_steps': n_steps,
        'n_assets': n_assets
    }

