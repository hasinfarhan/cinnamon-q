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
    
    Phase 5.1: Now includes Technical Analysis features (RSI, MACD, BBW, ATR, SMA Ratio).
    
    Args:
        data_path: Path to crypto_deep_v2.parquet (with TA features) or crypto_training_data.parquet (fallback)
        temp_dir: Temporary directory to save .npy files
    
    Returns:
        Dictionary with file paths and metadata:
            - 'returns_path': Path to returns_array.npy
            - 'volatility_path': Path to volatility_array.npy
            - 'commit_path': Path to commit_array.npy
            - 'utility_path': Path to utility_array.npy
            - 'rsi_path': Path to rsi_array.npy (NEW)
            - 'macd_path': Path to macd_array.npy (NEW)
            - 'bbw_path': Path to bbw_array.npy (NEW)
            - 'atr_path': Path to atr_array.npy (NEW)
            - 'sma_ratio_path': Path to sma_ratio_array.npy (NEW)
            - 'price_path': Path to price_array.npy
            - 'asset_list': List of asset symbols
            - 'timestamps': List of timestamps
            - 'n_steps': Number of timesteps
            - 'n_assets': Number of assets
            - 'has_ta_features': Boolean indicating if TA features are present
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Please run scripts/12_enrich_data.py to generate crypto_deep_v2.parquet with TA features."
        )
    
    print("📊 Loading and optimizing crypto training data (Main Process)...")
    
    # Load parquet file
    df = pd.read_parquet(data_path)
    df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
    
    # Check if TA features are present
    ta_columns = ['RSI_14', 'MACD', 'BB_WIDTH', 'ATR_NORM', 'SMA_200_RATIO']
    has_ta_features = all(col in df.columns for col in ta_columns)
    
    # Phase 5.5 Sprint 2: Check if correlation feature is present
    has_correlation = 'corr_btc' in df.columns
    
    if has_ta_features:
        print("  ✓ Technical Analysis features detected (RSI, MACD, BBW, ATR, SMA Ratio)")
    else:
        print("  ⚠ Technical Analysis features not found. Calculating basic features only.")
        print("     For best results, use crypto_deep_v2.parquet (run scripts/12_enrich_data.py)")
    
    if has_correlation:
        print("  ✓ Correlation feature detected (corr_btc - rolling correlation to BTC)")
    else:
        print("  ⚠ Correlation feature not found. Run scripts/12_enrich_data.py to add it.")
    
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
    
    # Phase 5.5 Sprint 2: Normalize correlation feature if present
    # Correlation is already in [-1, 1] range, but we can normalize to [0, 1] for consistency
    # Formula: (corr + 1) / 2 maps [-1, 1] -> [0, 1]
    if has_correlation:
        df['corr_btc_norm'] = ((df['corr_btc'] + 1.0) / 2.0).fillna(0.5).clip(0.0, 1.0)
        # 0.5 = neutral (correlation = 0), 1.0 = perfect positive, 0.0 = perfect negative
    else:
        # Default to 0.5 (neutral correlation) if not available
        df['corr_btc_norm'] = 0.5
    
    # Normalize TA features if present (per asset, vectorized)
    if has_ta_features:
        # RSI: Already 0-100, normalize to 0-1
        df['RSI_14_norm'] = df.groupby('ticker')['RSI_14'].transform(
            lambda x: (x - 0) / 100.0  # RSI is 0-100
        ).fillna(0.5).clip(0.0, 1.0)
        
        # MACD: Normalize per asset (min-max)
        macd_min = df.groupby('ticker')['MACD'].transform('min')
        macd_max = df.groupby('ticker')['MACD'].transform('max')
        macd_range = macd_max - macd_min
        # Use pandas where() instead of np.where() to keep it as a Series
        df['MACD_norm'] = pd.Series(
            np.where(
                macd_range > 0,
                (df['MACD'] - macd_min) / macd_range,
                0.0
            ),
            index=df.index
        ).fillna(0.0)
        
        # BB_WIDTH: Normalize per asset (min-max)
        bbw_min = df.groupby('ticker')['BB_WIDTH'].transform('min')
        bbw_max = df.groupby('ticker')['BB_WIDTH'].transform('max')
        bbw_range = bbw_max - bbw_min
        # Use pandas Series to keep it as a Series for fillna()
        df['BB_WIDTH_norm'] = pd.Series(
            np.where(
                bbw_range > 0,
                (df['BB_WIDTH'] - bbw_min) / bbw_range,
                0.0
            ),
            index=df.index
        ).fillna(0.0)
        
        # ATR_NORM: Already normalized, just ensure valid
        df['ATR_NORM'] = df['ATR_NORM'].fillna(0.0).clip(0.0, 1.0)
        
        # SMA_200_RATIO: Already a ratio, just ensure valid
        df['SMA_200_RATIO'] = df['SMA_200_RATIO'].fillna(1.0).clip(0.0, 2.0)  # Allow up to 2x
    
    print("  Pivoting data to wide format (vectorized reshape)...")
    
    # VECTORIZED Pivot: Convert from long format to wide format
    # This is the key optimization - replaces nested loops with C-level operations
    # Using pivot() instead of pivot_table() for better performance (no aggregation needed)
    
    # Pivot each feature separately
    # pivot() is faster than pivot_table() when we know there's one value per combination
    # If duplicates exist, pivot() will raise ValueError, so we catch and use pivot_table as fallback
    try:
        returns_pivot = df.pivot(index='timestamp', columns='ticker', values='returns')
        volatility_pivot = df.pivot(index='timestamp', columns='ticker', values='volatility')
        commit_pivot = df.pivot(index='timestamp', columns='ticker', values='commit_count_norm')
        utility_pivot = df.pivot(index='timestamp', columns='ticker', values='utility_score')
        price_pivot = df.pivot(index='timestamp', columns='ticker', values='close')
        
        # Phase 5.5 Sprint 2: Pivot correlation feature if present
        # Only create corr_pivot if correlation actually exists in the data
        if has_correlation:
            corr_pivot = df.pivot(index='timestamp', columns='ticker', values='corr_btc_norm')
        else:
            corr_pivot = None  # Don't create if not present
        
        # Pivot TA features if present
        if has_ta_features:
            rsi_pivot = df.pivot(index='timestamp', columns='ticker', values='RSI_14_norm')
            macd_pivot = df.pivot(index='timestamp', columns='ticker', values='MACD_norm')
            bbw_pivot = df.pivot(index='timestamp', columns='ticker', values='BB_WIDTH_norm')
            atr_pivot = df.pivot(index='timestamp', columns='ticker', values='ATR_NORM')
            sma_ratio_pivot = df.pivot(index='timestamp', columns='ticker', values='SMA_200_RATIO')
    except ValueError:
        # Fallback to pivot_table if duplicates exist (shouldn't happen, but safe)
        print("  Warning: Duplicates detected, using pivot_table() instead")
        returns_pivot = df.pivot_table(index='timestamp', columns='ticker', values='returns', aggfunc='first')
        volatility_pivot = df.pivot_table(index='timestamp', columns='ticker', values='volatility', aggfunc='first')
        commit_pivot = df.pivot_table(index='timestamp', columns='ticker', values='commit_count_norm', aggfunc='first')
        utility_pivot = df.pivot_table(index='timestamp', columns='ticker', values='utility_score', aggfunc='first')
        price_pivot = df.pivot_table(index='timestamp', columns='ticker', values='close', aggfunc='first')
        
        # Phase 5.5 Sprint 2: Pivot correlation feature if present
        if has_correlation:
            corr_pivot = df.pivot_table(index='timestamp', columns='ticker', values='corr_btc_norm', aggfunc='first')
        else:
            corr_pivot = None  # Don't create if not present
        
        if has_ta_features:
            rsi_pivot = df.pivot_table(index='timestamp', columns='ticker', values='RSI_14_norm', aggfunc='first')
            macd_pivot = df.pivot_table(index='timestamp', columns='ticker', values='MACD_norm', aggfunc='first')
            bbw_pivot = df.pivot_table(index='timestamp', columns='ticker', values='BB_WIDTH_norm', aggfunc='first')
            atr_pivot = df.pivot_table(index='timestamp', columns='ticker', values='ATR_NORM', aggfunc='first')
            sma_ratio_pivot = df.pivot_table(index='timestamp', columns='ticker', values='SMA_200_RATIO', aggfunc='first')
    
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
    
    # Phase 5.5 Sprint 2: Reindex correlation feature if present
    if has_correlation and corr_pivot is not None:
        corr_pivot = corr_pivot.reindex(index=common_index, columns=common_columns)
        corr_pivot = corr_pivot.ffill().fillna(0.5)  # Neutral correlation = 0.5 (maps to corr=0)
    
    # Reindex TA features if present
    if has_ta_features:
        rsi_pivot = rsi_pivot.reindex(index=common_index, columns=common_columns)
        rsi_pivot = rsi_pivot.ffill().fillna(0.5)  # RSI neutral = 50
        
        macd_pivot = macd_pivot.reindex(index=common_index, columns=common_columns)
        macd_pivot = macd_pivot.ffill().fillna(0.0)  # MACD neutral = 0
        
        bbw_pivot = bbw_pivot.reindex(index=common_index, columns=common_columns)
        bbw_pivot = bbw_pivot.ffill().fillna(0.0)
        
        atr_pivot = atr_pivot.reindex(index=common_index, columns=common_columns)
        atr_pivot = atr_pivot.ffill().fillna(0.0)
        
        sma_ratio_pivot = sma_ratio_pivot.reindex(index=common_index, columns=common_columns)
        sma_ratio_pivot = sma_ratio_pivot.ffill().fillna(1.0)  # SMA ratio neutral = 1.0
    
    # Convert to numpy arrays (maintaining [n_steps, n_assets] shape)
    returns_array = returns_pivot.values.astype(np.float32)
    volatility_array = volatility_pivot.values.astype(np.float32)
    commit_array = commit_pivot.values.astype(np.float32)
    utility_array = utility_pivot.values.astype(np.float32)
    price_array = price_pivot.values.astype(np.float32)
    
    # Phase 5.5 Sprint 2: Convert correlation array if present
    # Only create if correlation actually exists (for backward compatibility)
    if has_correlation and corr_pivot is not None:
        corr_array = corr_pivot.values.astype(np.float32)
    else:
        # Don't create array if correlation not present (backward compatibility)
        corr_array = None
    
    # Convert TA arrays if present
    if has_ta_features:
        rsi_array = rsi_pivot.values.astype(np.float32)
        macd_array = macd_pivot.values.astype(np.float32)
        bbw_array = bbw_pivot.values.astype(np.float32)
        atr_array = atr_pivot.values.astype(np.float32)
        sma_ratio_array = sma_ratio_pivot.values.astype(np.float32)
    else:
        # Create zero arrays for TA features (backward compatibility)
        rsi_array = np.zeros((n_steps, n_assets), dtype=np.float32)
        macd_array = np.zeros((n_steps, n_assets), dtype=np.float32)
        bbw_array = np.zeros((n_steps, n_assets), dtype=np.float32)
        atr_array = np.zeros((n_steps, n_assets), dtype=np.float32)
        sma_ratio_array = np.ones((n_steps, n_assets), dtype=np.float32)  # Neutral = 1.0
    
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
    
    # Phase 5.5 Sprint 2: Save correlation array
    corr_path = temp_dir / "corr_array.npy"
    np.save(str(corr_path), corr_array)
    
    # Save TA arrays if present
    if has_ta_features:
        rsi_path = temp_dir / "rsi_array.npy"
        macd_path = temp_dir / "macd_array.npy"
        bbw_path = temp_dir / "bbw_array.npy"
        atr_path = temp_dir / "atr_array.npy"
        sma_ratio_path = temp_dir / "sma_ratio_array.npy"
        
        np.save(str(rsi_path), rsi_array)
        np.save(str(macd_path), macd_array)
        np.save(str(bbw_path), bbw_array)
        np.save(str(atr_path), atr_array)
        np.save(str(sma_ratio_path), sma_ratio_array)
    
    # Calculate total memory usage
    total_bytes = (
        returns_array.nbytes +
        volatility_array.nbytes +
        commit_array.nbytes +
        utility_array.nbytes +
        price_array.nbytes +
        corr_array.nbytes  # Phase 5.5 Sprint 2: Correlation feature
    )
    if has_ta_features:
        total_bytes += (
            rsi_array.nbytes +
            macd_array.nbytes +
            bbw_array.nbytes +
            atr_array.nbytes +
            sma_ratio_array.nbytes
        )
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"✅ Data optimized and saved. Shape: [{n_steps}, {n_assets}]. Size: {total_mb:.2f} MB")
    print(f"   Files saved to: {temp_dir}")
    
    # Build return dictionary with all paths and metadata
    config = {
        'returns_path': str(returns_path),
        'volatility_path': str(volatility_path),
        'commit_path': str(commit_path),
        'utility_path': str(utility_path),
        'price_path': str(price_path),
        'corr_path': str(corr_path),  # Phase 5.5 Sprint 2: Correlation feature
        'asset_list': asset_list,
        'timestamps': timestamps,
        'n_steps': n_steps,
        'n_assets': n_assets,
        'has_ta_features': has_ta_features,  # CRITICAL: Include this flag!
        'has_correlation': has_correlation  # Phase 5.5 Sprint 2: Correlation flag
    }
    
    # Add TA paths if present
    if has_ta_features:
        config['rsi_path'] = str(rsi_path)
        config['macd_path'] = str(macd_path)
        config['bbw_path'] = str(bbw_path)
        config['atr_path'] = str(atr_path)
        config['sma_ratio_path'] = str(sma_ratio_path)
    
    return config

