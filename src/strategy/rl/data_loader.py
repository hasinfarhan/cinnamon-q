"""
Data Preprocessing for FastCryptoEnv - Load Once, Use Many (Memory-Mapped)

This module provides a function to preprocess the crypto training data
ONCE in the main process, then save it to disk for memory-mapped access
by multiple environment instances. This eliminates redundant file I/O,
processing, and pickling overhead when using SubprocVecEnv.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _fit_range(df: pd.DataFrame, col: str, fit_end: Optional[str], clip_pct: Optional[float] = None):
    """Per-asset (min, max) of `col`, fitted only on rows up to `fit_end` when it is given.

    With fit_end=None this equals groupby('ticker')[col].transform('min'/'max').
    v46: with clip_pct (e.g. 0.5) the range is the clip_pct / (100 - clip_pct) percentiles
    instead of min/max, so one extreme event does not set the scale; the caller clips the
    scaled value to [0, 1].
    """
    src = df if fit_end is None else df[pd.to_datetime(df['timestamp']) <= pd.Timestamp(fit_end)]
    if clip_pct:
        g = src.groupby('ticker')[col]
        stats = pd.DataFrame({'min': g.quantile(clip_pct / 100.0), 'max': g.quantile(1.0 - clip_pct / 100.0)})
    else:
        stats = src.groupby('ticker')[col].agg(['min', 'max'])
    return df['ticker'].map(stats['min']), df['ticker'].map(stats['max'])


def prepare_training_data(data_path: Path, temp_dir: Path, resample_freq: Optional[str] = None,
                          fit_end: Optional[str] = None, clip_pct: Optional[float] = None) -> Dict:
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
    
    print("[*] Loading and optimizing crypto training data (Main Process)...")

    # Load parquet file
    df = pd.read_parquet(data_path)
    df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)

    # Optional resampling to lower frequency (e.g. '4H' reduces 15-min bars to 4h bars)
    # Closes are taken as the last bar of each period; volumes summed; everything else last.
    if resample_freq is not None:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs = []
        for ticker, grp in df.groupby('ticker', sort=False):
            g = grp.set_index('timestamp')
            agg = {}
            for col in g.columns:
                cl = col.lower()
                if cl == 'open':            agg[col] = 'first'
                elif cl == 'high':          agg[col] = 'max'
                elif cl == 'low':           agg[col] = 'min'
                elif 'volume' in cl:        agg[col] = 'sum'
                else:                       agg[col] = 'last'
            r = g.resample(resample_freq).agg(agg).dropna(subset=['close'])
            r['ticker'] = ticker
            dfs.append(r.reset_index())
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        print(f"  Resampled to {resample_freq}: {len(df['timestamp'].unique()):,} timestamps")

    # Bars per day (used by fast_env for borrow rate and cooldown timing)
    _freq = (resample_freq or '').lower()
    if   _freq in ('4h',):  bars_per_day = 6
    elif _freq in ('1h',):  bars_per_day = 24
    elif _freq in ('1d',):  bars_per_day = 1
    else:                   bars_per_day = 96   # default: 15-min

    # Multi-timeframe window sizes scale with bar frequency
    if bars_per_day == 6:    # 4h bars
        mtf_windows = [(1, 'ret_4h', 0.02), (6, 'ret_1d', 0.05), (42, 'ret_7d', 0.15)]
        crypto_1d_w, crypto_7d_w = 6, 42
    elif bars_per_day == 24:  # 1h bars
        mtf_windows = [(4, 'ret_4h', 0.02), (24, 'ret_1d', 0.05), (168, 'ret_7d', 0.15)]
        crypto_1d_w, crypto_7d_w = 24, 168
    else:                     # 15-min bars (default)
        mtf_windows = [(16, 'ret_4h', 0.02), (96, 'ret_1d', 0.05), (672, 'ret_7d', 0.15)]
        crypto_1d_w, crypto_7d_w = 96, 672

    # Check if TA features are present
    ta_columns = ['RSI_14', 'MACD', 'BB_WIDTH', 'ATR_NORM', 'SMA_200_RATIO']
    has_ta_features = all(col in df.columns for col in ta_columns)
    
    # Phase 5.5 Sprint 2: Check if correlation feature is present
    has_correlation = 'corr_btc' in df.columns
    
    # Phase 6 Sprint 1: Check if GARCH volatility feature is present
    has_garch = 'garch_volatility' in df.columns

    # Macro features: market-wide context (same value across all assets per timestamp)
    MACRO_COLS = ['vix', 'dxy', 'tnx', 'macro_trend_score']
    has_macro = all(c in df.columns for c in MACRO_COLS)
    
    if has_ta_features:
        print("  [OK] Technical Analysis features detected (RSI, MACD, BBW, ATR, SMA Ratio)")
    else:
        print("  [!!] Technical Analysis features not found. Calculating basic features only.")
        print("     For best results, use crypto_deep_v2.parquet (run scripts/12_enrich_data.py)")

    if has_correlation:
        print("  [OK] Correlation feature detected (corr_btc - rolling correlation to BTC)")
    else:
        print("  [!!] Correlation feature not found. Run scripts/12_enrich_data.py to add it.")

    if has_garch:
        print("  [OK] GARCH volatility feature detected (predictive risk - Phase 6)")
    else:
        print("  [!!] GARCH volatility not found. Run scripts/12_enrich_data.py to add it.")

    if has_macro:
        print("  [OK] Macro features detected (VIX, DXY, TNX, macro_trend_score)")
    else:
        print("  [!!] Macro features not found (vix/dxy/tnx/macro_trend_score). Run scripts/16_fill_macro_features.py.")
    
    # Get unique assets and timestamps
    asset_list = sorted(df['ticker'].unique().tolist())
    timestamps = sorted(df['timestamp'].unique())
    n_steps = len(timestamps)
    n_assets = len(asset_list)
    
    print(f"  Found {n_steps} timestamps, {n_assets} assets")
    
    # Phase 6 Sprint 2: Extract BTC benchmark returns for DIR calculation
    benchmark_ticker = 'BTC'  # Use BTC as benchmark
    benchmark_returns = None
    if benchmark_ticker in asset_list:
        btc_data = df[df['ticker'] == benchmark_ticker].sort_values('timestamp')
        if len(btc_data) > 0:
            # Calculate returns (pct_change)
            btc_returns = btc_data.groupby('timestamp')['close'].first().pct_change().fillna(0.0)
            # Align with timestamps
            benchmark_returns = np.zeros(n_steps, dtype=np.float32)
            for i, ts in enumerate(timestamps):
                if ts in btc_returns.index:
                    benchmark_returns[i] = float(btc_returns[ts])
            print(f"  [OK] Benchmark returns extracted: BTC/USD ({benchmark_returns.shape})")
        else:
            print(f"  [!!] Warning: BTC data not found, DIR will use zero benchmark")
            benchmark_returns = np.zeros(n_steps, dtype=np.float32)
    else:
        print(f"  [!!] Warning: BTC not in asset list, DIR will use zero benchmark")
        benchmark_returns = np.zeros(n_steps, dtype=np.float32)
    
    print("  Calculating features using vectorized operations...")
    
    # Sort by ticker and timestamp for efficient groupby operations
    df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # VECTORIZED Feature Engineering (Much faster than loops!)
    # Calculate returns per asset using groupby (vectorized)
    df['returns'] = df.groupby('ticker')['close'].pct_change().fillna(0.0)

    # ── Multi-timeframe cumulative returns (Problem 1 fix) ─────────────────
    # pct_change(N) = (price[t] / price[t-N]) - 1  →  cumulative return over N bars
    # Window sizes are scaled to the bar frequency (mtf_windows computed above).
    # Normalised with tanh so large moves are compressed: ±scale → ±0.76
    for window, col, scale in mtf_windows:
        raw = df.groupby('ticker')['close'].pct_change(periods=window).fillna(0.0)
        df[col + '_norm'] = np.tanh(raw / scale).astype(np.float32)

    # Calculate volatility (rolling std) per asset using groupby (vectorized)
    df['volatility'] = df.groupby('ticker')['returns'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    ).fillna(0.0)
    
    # Normalize commit_count per asset (vectorized)
    # Group by ticker, calculate min/max, then normalize
    if fit_end is not None:
        print(f"  [HOLDOUT] min-max normalization ranges fitted on rows up to {fit_end} only")
    commit_min, commit_max = _fit_range(df, 'commit_count_4_weeks', fit_end)
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
        macd_min, macd_max = _fit_range(df, 'MACD', fit_end, clip_pct)
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
        if clip_pct:
            df['MACD_norm'] = df['MACD_norm'].clip(0.0, 1.0)
        
        # BB_WIDTH: Normalize per asset (min-max)
        bbw_min, bbw_max = _fit_range(df, 'BB_WIDTH', fit_end, clip_pct)
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
        if clip_pct:
            df['BB_WIDTH_norm'] = df['BB_WIDTH_norm'].clip(0.0, 1.0)
        
        # ATR_NORM: Already normalized, just ensure valid
        df['ATR_NORM'] = df['ATR_NORM'].fillna(0.0).clip(0.0, 1.0)
        
        # SMA_200_RATIO: normalize to [0, 1]. Raw value is price/SMA200 (typically 0.5–1.5).
        # Clip to [0, 2] then divide by 2 → [0, 1]. 1.0 = at SMA, 0.5 = 2× above SMA.
        df['SMA_200_RATIO'] = (df['SMA_200_RATIO'].fillna(1.0).clip(0.0, 2.0) / 2.0)
    
    # Phase 6 Sprint 1: Normalize GARCH volatility if present (per asset, vectorized)
    # GARCH volatility is already in a reasonable range (typically 0.001-0.1), but normalize per asset
    if has_garch:
        # Normalize GARCH volatility per asset (min-max scaling)
        garch_min, garch_max = _fit_range(df, 'garch_volatility', fit_end, clip_pct)
        garch_range = garch_max - garch_min
        df['garch_volatility_norm'] = pd.Series(
            np.where(
                garch_range > 0,
                (df['garch_volatility'] - garch_min) / garch_range,
                0.5  # Default if no range
            ),
            index=df.index
        ).fillna(0.5).clip(0.0, 1.0)
    else:
        # Default to 0.5 (neutral volatility) if not available
        df['garch_volatility_norm'] = 0.5
    
    # ── Macro features (global context, one value per timestamp) ──────────────
    # Extract from any ticker (all have the same value per timestamp).
    # Normalization bounds match scripts/16_fill_macro_features.py.
    if has_macro:
        macro_ts = (
            df.sort_values('timestamp')
              .groupby('timestamp')[MACRO_COLS]
              .first()
              .reindex(pd.Index(timestamps, name='timestamp'))
              .ffill()
              .bfill()
        )
        VIX_LO, VIX_HI   = 10.0, 80.0
        DXY_LO, DXY_HI   = 88.0, 115.0
        TNX_LO, TNX_HI   =  0.5,   5.5

        vix_norm   = np.clip((macro_ts['vix'].values   - VIX_LO) / (VIX_HI - VIX_LO), 0, 1)
        dxy_norm   = np.clip((macro_ts['dxy'].values   - DXY_LO) / (DXY_HI - DXY_LO), 0, 1)
        tnx_norm   = np.clip((macro_ts['tnx'].values   - TNX_LO) / (TNX_HI - TNX_LO), 0, 1)
        macro_score = np.clip(macro_ts['macro_trend_score'].values, 0, 1)

        # Shape: [n_steps, 4]  (VIX, DXY, TNX, macro_trend_score)
        macro_array = np.stack([vix_norm, dxy_norm, tnx_norm, macro_score], axis=1).astype(np.float32)
        print(f"  [OK] Macro array built: shape={macro_array.shape}  "
              f"VIX mean={vix_norm.mean():.2f}  macro_score mean={macro_score.mean():.2f}")
    else:
        macro_array = None

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
        
        # Phase 6 Sprint 1: Pivot GARCH volatility if present
        if has_garch:
            garch_pivot = df.pivot(index='timestamp', columns='ticker', values='garch_volatility_norm')
        else:
            garch_pivot = None  # Don't create if not present
        
        # Pivot TA features if present
        if has_ta_features:
            rsi_pivot = df.pivot(index='timestamp', columns='ticker', values='RSI_14_norm')
            macd_pivot = df.pivot(index='timestamp', columns='ticker', values='MACD_norm')
            bbw_pivot = df.pivot(index='timestamp', columns='ticker', values='BB_WIDTH_norm')
            atr_pivot = df.pivot(index='timestamp', columns='ticker', values='ATR_NORM')
            sma_ratio_pivot = df.pivot(index='timestamp', columns='ticker', values='SMA_200_RATIO')

        # Multi-timeframe return pivots (always present after v7)
        ret_4h_pivot = df.pivot(index='timestamp', columns='ticker', values='ret_4h_norm')
        ret_1d_pivot = df.pivot(index='timestamp', columns='ticker', values='ret_1d_norm')
        ret_7d_pivot = df.pivot(index='timestamp', columns='ticker', values='ret_7d_norm')
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
        
        # Phase 6 Sprint 1: Pivot GARCH volatility if present
        if has_garch:
            garch_pivot = df.pivot_table(index='timestamp', columns='ticker', values='garch_volatility_norm', aggfunc='first')
        else:
            garch_pivot = None  # Don't create if not present
        
        if has_ta_features:
            rsi_pivot = df.pivot_table(index='timestamp', columns='ticker', values='RSI_14_norm', aggfunc='first')
            macd_pivot = df.pivot_table(index='timestamp', columns='ticker', values='MACD_norm', aggfunc='first')
            bbw_pivot = df.pivot_table(index='timestamp', columns='ticker', values='BB_WIDTH_norm', aggfunc='first')
            atr_pivot = df.pivot_table(index='timestamp', columns='ticker', values='ATR_NORM', aggfunc='first')
            sma_ratio_pivot = df.pivot_table(index='timestamp', columns='ticker', values='SMA_200_RATIO', aggfunc='first')

        ret_4h_pivot = df.pivot_table(index='timestamp', columns='ticker', values='ret_4h_norm', aggfunc='first')
        ret_1d_pivot = df.pivot_table(index='timestamp', columns='ticker', values='ret_1d_norm', aggfunc='first')
        ret_7d_pivot = df.pivot_table(index='timestamp', columns='ticker', values='ret_7d_norm', aggfunc='first')
    
    # Ensure all pivots have the same index (timestamps) and columns (assets)
    # Reindex to ensure alignment and handle missing values
    common_index = pd.Index(timestamps, name='timestamp')
    common_columns = pd.Index(asset_list, name='ticker')
    
    # Reindex and fill missing values
    # Use ffill (forward fill) to handle gaps, then fill remaining NaNs
    returns_pivot = returns_pivot.reindex(index=common_index, columns=common_columns)
    # v46: a missing bar has no return (was ffill, which repeated the previous bar's return;
    # identical for v5, whose pivots have no holes)
    returns_pivot = returns_pivot.fillna(0.0)
    
    volatility_pivot = volatility_pivot.reindex(index=common_index, columns=common_columns)
    volatility_pivot = volatility_pivot.ffill().fillna(0.0)
    
    commit_pivot = commit_pivot.reindex(index=common_index, columns=common_columns)
    commit_pivot = commit_pivot.ffill().fillna(0.5)
    
    utility_pivot = utility_pivot.reindex(index=common_index, columns=common_columns)
    utility_pivot = utility_pivot.ffill().fillna(0.5)
    
    price_pivot = price_pivot.reindex(index=common_index, columns=common_columns)
    # v46: listing mask. A coin is tradable once it has 20 days of its own price history
    # (the semicovariance window); a coin present from the first bar is tradable at once,
    # so a dataset where every coin starts together (v5) gets an all-True mask.
    _first = price_pivot.notna().values.argmax(axis=0)            # first listed bar per coin
    _lag = np.where(_first > 0, bars_per_day * 20, 0)
    avail_array = (np.arange(n_steps)[:, None] >= (_first + _lag)[None, :])
    for _a, _f in zip(asset_list, _first):
        if _f > 0:
            print(f"  [LISTING] {_a}: first bar {timestamps[_f]}, tradable from bar {_f + bars_per_day * 20}")
    price_pivot = price_pivot.ffill().fillna(0.0)
    _price_listed = price_pivot.where(price_pivot > 0)              # NaN before listing
    
    # Phase 5.5 Sprint 2: Reindex correlation feature if present
    if has_correlation and corr_pivot is not None:
        corr_pivot = corr_pivot.reindex(index=common_index, columns=common_columns)
        corr_pivot = corr_pivot.ffill().fillna(0.5)  # Neutral correlation = 0.5 (maps to corr=0)
    
    # Phase 6 Sprint 1: Reindex GARCH volatility if present
    if has_garch and garch_pivot is not None:
        garch_pivot = garch_pivot.reindex(index=common_index, columns=common_columns)
        garch_pivot = garch_pivot.ffill().fillna(0.5)  # Neutral volatility = 0.5
    
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

    # Multi-timeframe returns: reindex + neutral fill (0.0 = no move)
    ret_4h_pivot = ret_4h_pivot.reindex(index=common_index, columns=common_columns).ffill().fillna(0.0)
    ret_1d_pivot = ret_1d_pivot.reindex(index=common_index, columns=common_columns).ffill().fillna(0.0)
    ret_7d_pivot = ret_7d_pivot.reindex(index=common_index, columns=common_columns).ffill().fillna(0.0)

    # ── Crypto regime features (Problem 2 fix) ────────────────────────────
    # Four global signals that directly reflect the crypto market trend.
    # These are crypto-native equivalents of VIX/DXY/TNX which only capture
    # traditional market risk — they miss crypto-specific bear markets.
    #   [0] BTC 1d return  — short-term trend (tanh-normalised)
    #   [1] BTC 7d return  — medium-term bear/bull direction
    #   [2] Market 7d return — equal-weight avg of all 14 assets (broad crypto market)
    #   [3] BTC vs SMA200  — structural trend (0=far below, 0.5=at SMA, 1=far above)
    if 'BTC' in price_pivot.columns:
        btc_1d_ret = price_pivot['BTC'].pct_change(crypto_1d_w).fillna(0.0).values
        btc_7d_ret = price_pivot['BTC'].pct_change(crypto_7d_w).fillna(0.0).values
    else:
        btc_1d_ret = np.zeros(n_steps, dtype=np.float32)
        btc_7d_ret = np.zeros(n_steps, dtype=np.float32)

    # v46: average over listed coins only (price 0 before listing made pct_change inf)
    market_7d_ret = _price_listed.pct_change(crypto_7d_w, fill_method=None).mean(axis=1).fillna(0.0).values

    # Compute true 200-day SMA ratio from 4h price data (1200 bars = 200 days x 6 bars/day).
    # The parquet SMA_200_RATIO is a 200-bar 15-min (50-hour) SMA — price rarely deviates from
    # it, so after normalization it's always ~0.5 (neutral), making sma_sig always ~0 and
    # rendering direction-lock, C4, C6 all useless. We need the structural 200-DAY trend.
    # Normalization: clip(price/SMA200d, 0.5, 1.5) → subtract 0.5 → [0, 1], 0.5=at SMA.
    if 'BTC' in price_pivot.columns:
        btc_price_4h = price_pivot['BTC']
        btc_sma200_day = btc_price_4h.rolling(window=1200, min_periods=100).mean().bfill()
        raw_ratio = (btc_price_4h / btc_sma200_day).clip(0.5, 1.5)
        btc_sma200 = ((raw_ratio - 0.5) / 1.0).values.astype(np.float32)  # [0,1], 0.5=at SMA
    else:
        btc_sma200 = np.full(n_steps, 0.5, dtype=np.float32)

    # v41: Per-asset 200-day SMA ratio — same normalization as BTC above.
    # Gives C4/C6 per-asset signals so the model learns cross-sectional allocation:
    # "SOL above its 200d MA → reward long SOL" vs "BAT below → reward short BAT".
    # Without this, C6 used one BTC signal for all 14 assets, teaching only macro direction.
    asset_sma200_cols = []
    for col in price_pivot.columns:
        col_price = _price_listed[col]
        # v49: a true 200-day average (full 1,200 bars), no back-fill (bfill copied later SMA values
        # into earlier bars); until a coin has 200 days of history its trend reads neutral (0.5)
        col_sma200 = col_price.rolling(window=1200, min_periods=1200).mean()
        col_ratio  = (col_price / col_sma200).clip(0.5, 1.5)
        col_norm   = ((col_ratio - 0.5) / 1.0).fillna(0.5).values.astype(np.float32)
        asset_sma200_cols.append(col_norm)
    asset_sma200_array = np.stack(asset_sma200_cols, axis=1)  # (n_steps, n_assets)

    crypto_regime_array = np.stack([
        np.tanh(btc_1d_ret   / 0.05).astype(np.float32),   # BTC 1d
        np.tanh(btc_7d_ret   / 0.15).astype(np.float32),   # BTC 7d
        np.tanh(market_7d_ret/ 0.15).astype(np.float32),   # Market 7d
        np.clip(btc_sma200, 0.0, 1.0),                      # BTC vs SMA200
    ], axis=1)
    print(f"  [OK] Crypto regime array: shape={crypto_regime_array.shape}  "
          f"BTC_7d mean={crypto_regime_array[:,1].mean():.3f}  "
          f"btc_sma200 mean={crypto_regime_array[:,3].mean():.3f}")

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
    
    # Phase 6 Sprint 1: Convert GARCH volatility array if present
    if has_garch and garch_pivot is not None:
        garch_array = garch_pivot.values.astype(np.float32)
    else:
        # Don't create array if GARCH not present (backward compatibility)
        garch_array = None
    
    # Multi-timeframe return arrays (always present v7+)
    ret_4h_array = ret_4h_pivot.values.astype(np.float32)
    ret_1d_array = ret_1d_pivot.values.astype(np.float32)
    ret_7d_array = ret_7d_pivot.values.astype(np.float32)

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
    benchmark_path = temp_dir / "benchmark_returns.npy"  # Phase 6 Sprint 2: DIR benchmark
    
    np.save(str(returns_path), returns_array)
    np.save(str(volatility_path), volatility_array)
    np.save(str(commit_path), commit_array)
    np.save(str(utility_path), utility_array)
    np.save(str(price_path), price_array)
    np.save(str(benchmark_path), benchmark_returns)  # Phase 6 Sprint 2: Save benchmark returns

    # Macro features
    macro_path = temp_dir / "macro_array.npy"
    if macro_array is not None:
        np.save(str(macro_path), macro_array)
    else:
        np.save(str(macro_path), np.full((n_steps, 4), 0.5, dtype=np.float32))

    # Multi-timeframe returns (v7+)
    ret_4h_path  = temp_dir / "ret_4h_array.npy"
    ret_1d_path  = temp_dir / "ret_1d_array.npy"
    ret_7d_path  = temp_dir / "ret_7d_array.npy"
    np.save(str(ret_4h_path),  ret_4h_array)
    np.save(str(ret_1d_path),  ret_1d_array)
    np.save(str(ret_7d_path),  ret_7d_array)

    # Crypto regime features (v7+)
    crypto_regime_path = temp_dir / "crypto_regime_array.npy"
    np.save(str(crypto_regime_path), crypto_regime_array)
    
    # Phase 5.5 Sprint 2: Save correlation array (only if present)
    corr_path = temp_dir / "corr_array.npy"
    if corr_array is not None:
        np.save(str(corr_path), corr_array)
    else:
        # Create dummy array for backward compatibility (won't be used)
        np.save(str(corr_path), np.zeros((n_steps, n_assets), dtype=np.float32))
    
    # Phase 6 Sprint 1: Save GARCH volatility array (only if present)
    garch_path = temp_dir / "garch_array.npy"
    if garch_array is not None:
        np.save(str(garch_path), garch_array)
    else:
        # Create dummy array for backward compatibility (won't be used)
        np.save(str(garch_path), np.full((n_steps, n_assets), 0.5, dtype=np.float32))
    
    # v41: Per-asset 200-day SMA ratio array (always saved — required for C4/C6)
    asset_sma200_path = temp_dir / "asset_sma200_array.npy"
    np.save(str(asset_sma200_path), asset_sma200_array)
    avail_path = temp_dir / "avail_array.npy"
    np.save(str(avail_path), avail_array)

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
        price_array.nbytes
    )
    if corr_array is not None:
        total_bytes += corr_array.nbytes  # Phase 5.5 Sprint 2: Correlation feature
    if garch_array is not None:
        total_bytes += garch_array.nbytes  # Phase 6 Sprint 1: GARCH volatility feature
    if has_ta_features:
        total_bytes += (
            rsi_array.nbytes +
            macd_array.nbytes +
            bbw_array.nbytes +
            atr_array.nbytes +
            sma_ratio_array.nbytes
        )
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"[OK] Data optimized and saved. Shape: [{n_steps}, {n_assets}]. Size: {total_mb:.2f} MB")
    print(f"   Files saved to: {temp_dir}")
    
    # Build return dictionary with all paths and metadata
    config = {
        'returns_path': str(returns_path),
        'volatility_path': str(volatility_path),
        'commit_path': str(commit_path),
        'utility_path': str(utility_path),
        'price_path': str(price_path),
        'corr_path': str(corr_path),  # Phase 5.5 Sprint 2: Correlation feature
        'garch_path': str(garch_path),  # Phase 6 Sprint 1: GARCH volatility feature
        'benchmark_path': str(benchmark_path),  # Phase 6 Sprint 2: DIR benchmark returns
        'asset_list': asset_list,
        'timestamps': timestamps,
        'n_steps': n_steps,
        'n_assets': n_assets,
        'has_ta_features': has_ta_features,  # CRITICAL: Include this flag!
        'has_correlation': has_correlation,  # Phase 5.5 Sprint 2: Correlation flag
        'has_garch': has_garch,  # Phase 6 Sprint 1: GARCH volatility flag
        'has_macro': has_macro,  # Macro context features (VIX, DXY, TNX, macro_trend_score)
        'macro_path': str(macro_path),
        # v7: multi-timeframe returns + crypto regime
        'has_mtf': True,
        'ret_4h_path': str(ret_4h_path),
        'ret_1d_path': str(ret_1d_path),
        'ret_7d_path': str(ret_7d_path),
        'crypto_regime_path': str(crypto_regime_path),
        # v27: bar frequency metadata consumed by fast_env for rate/cooldown scaling
        'bars_per_day': bars_per_day,
        # v41: per-asset 200-day SMA ratio for C4/C6 cross-sectional signals
        'asset_sma200_path': str(asset_sma200_path),
        # v46: [n_steps, n_assets] bool, True once a coin is listed and tradable
        'avail_path': str(avail_path),
    }

    # Add TA paths if present
    if has_ta_features:
        config['rsi_path'] = str(rsi_path)
        config['macd_path'] = str(macd_path)
        config['bbw_path'] = str(bbw_path)
        config['atr_path'] = str(atr_path)
        config['sma_ratio_path'] = str(sma_ratio_path)
    
    return config

