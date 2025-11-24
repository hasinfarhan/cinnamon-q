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
    print("  Calculating features using vectorized operations...")
    
    # Sort by ticker and timestamp for efficient groupby operations
    df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # VECTORIZED Feature Engineering (Much faster than loops!)
    # Calculate returns per asset using groupby (vectorized)
    df['returns'] = df.groupby('ticker')['close'].pct_change().fillna(0.0)
    
    # Calculate volatility (rolling std) per asset using groupby (vectorized)
    df['volatility'] = df.groupby('ticker')['returns'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    ).fillna(0.0)
    
    # Normalize commit_count per asset (vectorized)
    # Group by ticker, calculate min/max, then normalize
    commit_min = df.groupby('ticker')['commit_count_4_weeks'].transform('min')
    commit_max = df.groupby('ticker')['commit_count_4_weeks'].transform('max')
    commit_range = commit_max - commit_min
    df['commit_count_norm'] = np.where(
        commit_range > 0,
        (df['commit_count_4_weeks'] - commit_min) / commit_range,
        0.5  # Default if no range
    )
    
    # Ensure utility_score is valid (vectorized)
    df['utility_score'] = df['utility_score'].fillna(0.5).clip(0.0, 1.0)
    
    print("  Pivoting data to wide format (vectorized reshape)...")
    
    # VECTORIZED Pivot: Convert from long format to wide format
    # This is the key optimization - replaces nested loops with C-level operations
    # Using pivot() instead of pivot_table() for better performance (no aggregation needed)
    
    # Create a complete index/column structure first
    # This ensures all timestamp-ticker combinations exist
    complete_index = pd.MultiIndex.from_product(
        [timestamps, asset_list],
        names=['timestamp', 'ticker']
    )
    
    # Pivot each feature separately
    # pivot() is faster than pivot_table() when we know there's one value per combination
    # If duplicates exist, pivot() will raise ValueError, so we catch and use pivot_table as fallback
    try:
        returns_pivot = df.pivot(index='timestamp', columns='ticker', values='returns')
        volatility_pivot = df.pivot(index='timestamp', columns='ticker', values='volatility')
        commit_pivot = df.pivot(index='timestamp', columns='ticker', values='commit_count_norm')
        utility_pivot = df.pivot(index='timestamp', columns='ticker', values='utility_score')
        price_pivot = df.pivot(index='timestamp', columns='ticker', values='close')
    except ValueError:
        # Fallback to pivot_table if duplicates exist (shouldn't happen, but safe)
        print("  Warning: Duplicates detected, using pivot_table() instead")
        returns_pivot = df.pivot_table(index='timestamp', columns='ticker', values='returns', aggfunc='first')
        volatility_pivot = df.pivot_table(index='timestamp', columns='ticker', values='volatility', aggfunc='first')
        commit_pivot = df.pivot_table(index='timestamp', columns='ticker', values='commit_count_norm', aggfunc='first')
        utility_pivot = df.pivot_table(index='timestamp', columns='ticker', values='utility_score', aggfunc='first')
        price_pivot = df.pivot_table(index='timestamp', columns='ticker', values='close', aggfunc='first')
    
    # Ensure all pivots have the same index (timestamps) and columns (assets)
    # Reindex to ensure alignment and handle missing values
    common_index = pd.Index(timestamps, name='timestamp')
    common_columns = pd.Index(asset_list, name='ticker')
    
    # Reindex and fill missing values
    # Use ffill (forward fill) to handle gaps, then fill remaining NaNs
    returns_pivot = returns_pivot.reindex(index=common_index, columns=common_columns)
    returns_pivot = returns_pivot.ffill().fillna(0.0)  # Forward fill then zero
    
    volatility_pivot = volatility_pivot.reindex(index=common_index, columns=common_columns)
    volatility_pivot = volatility_pivot.ffill().fillna(0.0)
    
    commit_pivot = commit_pivot.reindex(index=common_index, columns=common_columns)
    commit_pivot = commit_pivot.ffill().fillna(0.5)
    
    utility_pivot = utility_pivot.reindex(index=common_index, columns=common_columns)
    utility_pivot = utility_pivot.ffill().fillna(0.5)
    
    price_pivot = price_pivot.reindex(index=common_index, columns=common_columns)
    price_pivot = price_pivot.ffill().fillna(0.0)
    
    # Convert to numpy arrays (maintaining [n_steps, n_assets] shape)
    returns_array = returns_pivot.values.astype(np.float32)
    volatility_array = volatility_pivot.values.astype(np.float32)
    commit_array = commit_pivot.values.astype(np.float32)
    utility_array = utility_pivot.values.astype(np.float32)
    price_array = price_pivot.values.astype(np.float32)
    
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

