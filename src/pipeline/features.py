"""
Feature Engineering Stage - Add Technical Analysis Indicators

This module wraps the feature engineering logic from scripts/12_enrich_data.py
into a callable function for the MLOps pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from features.indicators import TechnicalAnalysis

logger = logging.getLogger(__name__)


def run_feature_engineering(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    Run the feature engineering stage.
    
    Enriches the input dataset with technical analysis indicators:
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Band Width
    - ATR (Normalized)
    - SMA Ratio (Close/SMA200)
    
    Args:
        input_path: Path to input parquet file (default: data/processed/crypto_deep_master.parquet)
        output_path: Path to save enriched data (default: data/processed/crypto_deep_v2.parquet)
    
    Returns:
        Path to the saved enriched parquet file
    """
    logger.info("=" * 80)
    logger.info("Stage 2: Feature Engineering")
    logger.info("=" * 80)
    
    project_root = Path(__file__).parent.parent.parent
    
    if input_path is None:
        input_path = project_root / "data" / "processed" / "crypto_deep_master.parquet"
    
    if output_path is None:
        output_path = project_root / "data" / "processed" / "crypto_deep_v2.parquet"
    
    # Load data
    logger.info(f"Loading data from {input_path}...")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please run ingestion stage first."
        )
    
    df = pd.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} rows, {df['ticker'].nunique()} assets")
    
    # Initialize TA calculator
    logger.info("Initializing Technical Analysis calculator...")
    ta = TechnicalAnalysis()
    
    # Add features
    logger.info("Adding technical indicators (per ticker)...")
    df_enriched = ta.add_all_features(df)
    
    # Verify new columns
    original_cols = set(df.columns)
    new_cols = set(df_enriched.columns) - original_cols
    logger.info(f"  Added {len(new_cols)} new feature columns")
    
    # Final cleanup: ensure no NaNs
    feature_cols = ta._get_feature_columns()
    for col in feature_cols:
        if col in df_enriched.columns:
            if 'RATIO' in col:
                df_enriched[col] = df_enriched[col].fillna(1.0)
            else:
                df_enriched[col] = df_enriched[col].fillna(0.0)
    
    # Save enriched data
    logger.info(f"Saving enriched data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(output_path, index=False, compression='snappy')
    
    logger.info(f"✓ Feature engineering complete: {len(df_enriched):,} rows, "
               f"{len(df_enriched.columns)} columns, "
               f"{output_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    return output_path

