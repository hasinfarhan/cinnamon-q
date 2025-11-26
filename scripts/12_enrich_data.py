"""
Data Enrichment Script - Deep Feature Engineering

This script enriches the crypto training data with professional technical analysis
indicators (RSI, MACD, Bollinger Bands, ATR, SMA Ratio) to give the AI better "eyes"
for understanding market patterns.

Phase 3.5: Deep Feature Engineering
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from features.indicators import TechnicalAnalysis


def main():
    """Main execution function."""
    print("=" * 80)
    print("🔬 Deep Feature Engineering - Data Enrichment")
    print("Phase 3.5: Adding Technical Analysis Indicators")
    print("=" * 80)
    print()
    
    # Paths
    # Use crypto_training_data.parquet as input (has utility_score from sentiment analysis)
    # Fallback to crypto_deep_master.parquet if training data doesn't exist
    training_data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
    master_data_path = project_root / "data" / "processed" / "crypto_deep_master.parquet"
    output_path = project_root / "data" / "processed" / "crypto_deep_v2.parquet"
    
    # Step 1: Load existing data
    print("Step 1: Loading existing data...")
    if training_data_path.exists():
        input_path = training_data_path
        print(f"  ✓ Using crypto_training_data.parquet (includes sentiment/utility_score)")
    elif master_data_path.exists():
        input_path = master_data_path
        print(f"  ⚠ Using crypto_deep_master.parquet (no utility_score - will use neutral 0.5)")
    else:
        raise FileNotFoundError(
            f"Input file not found: {training_data_path} or {master_data_path}\n"
            "Please run scripts/08_fetch_crypto_deep.py and scripts/09_generate_sentiment.py first."
        )
    
    df = pd.read_parquet(input_path)
    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  ✓ Assets: {df['ticker'].nunique()}")
    print(f"  ✓ Columns: {len(df.columns)} total")
    
    # Ensure utility_score exists (required for training)
    if 'utility_score' not in df.columns:
        print(f"  ⚠ Warning: utility_score not found, adding neutral values (0.5)")
        df['utility_score'] = 0.5
    
    # Ensure commit_count_4_weeks exists (required for training)
    if 'commit_count_4_weeks' not in df.columns:
        print(f"  ⚠ Warning: commit_count_4_weeks not found, adding placeholder values (100.0)")
        df['commit_count_4_weeks'] = 100.0
    
    print()
    
    # Step 2: Initialize Technical Analysis calculator
    print("Step 2: Initializing Technical Analysis calculator...")
    ta = TechnicalAnalysis()
    print("  ✓ TechnicalAnalysis initialized")
    print()
    
    # Step 3: Add technical indicators
    print("Step 3: Adding technical indicators (per ticker)...")
    print("  This may take a few minutes for large datasets...")
    df_enriched = ta.add_all_features(df)
    print()
    
    # Step 4: Verify new columns
    print("Step 4: Verifying enriched data...")
    original_cols = set(df.columns)
    new_cols = set(df_enriched.columns) - original_cols
    print(f"  ✓ Original columns: {len(original_cols)}")
    print(f"  ✓ New columns: {len(new_cols)}")
    print(f"  ✓ Total columns: {len(df_enriched.columns)}")
    print()
    print("  New feature columns:")
    for col in sorted(new_cols):
        non_null_count = df_enriched[col].notna().sum()
        null_count = df_enriched[col].isna().sum()
        print(f"    - {col:20s}: {non_null_count:,} non-null, {null_count:,} null")
    print()
    
    # Step 5: Data quality check
    print("Step 5: Data quality check...")
    # Check for any remaining NaNs in new columns
    feature_cols = ta._get_feature_columns()
    nan_counts = {}
    for col in feature_cols:
        if col in df_enriched.columns:
            nan_count = df_enriched[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = nan_count
    
    if nan_counts:
        print(f"  ⚠ Warning: Found NaNs in feature columns:")
        for col, count in nan_counts.items():
            print(f"    - {col}: {count:,} NaNs")
        print("  Attempting to fill remaining NaNs...")
        # Final cleanup: fill any remaining NaNs
        for col in feature_cols:
            if col in df_enriched.columns:
                if 'RATIO' in col:
                    df_enriched[col] = df_enriched[col].fillna(1.0)
                else:
                    df_enriched[col] = df_enriched[col].fillna(0.0)
        print("  ✓ NaNs filled")
    else:
        print("  ✓ No NaNs found in feature columns")
    print()
    
    # Step 6: Save enriched data
    print("Step 6: Saving enriched data...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(output_path, index=False, compression='snappy')
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    print()
    
    # Step 7: Summary statistics
    print("Step 7: Feature summary statistics...")
    print("  Sample statistics for new features (first ticker):")
    first_ticker = df_enriched['ticker'].iloc[0]
    ticker_data = df_enriched[df_enriched['ticker'] == first_ticker]
    
    for col in feature_cols:
        if col in ticker_data.columns:
            stats = ticker_data[col].describe()
            print(f"    {col:20s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"min={stats['min']:.4f}, max={stats['max']:.4f}")
    print()
    
    print("=" * 80)
    print("✅ Data enrichment complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review the enriched data: data/processed/crypto_deep_v2.parquet")
    print("  2. Update training scripts to use the new features")
    print("  3. Retrain the model with enriched features (v2.0 training run)")
    print()


if __name__ == "__main__":
    main()

