"""
Generate FinBERT Embeddings for News Headlines

Research Track Sprint 1: Semantic Grounding with Embeddings

This script generates 768-dimensional FinBERT embeddings from news headlines
and saves them in a time-aligned format compatible with the training data.

The embeddings are stored as:
- data/processed/news_embeddings.npy: Numpy array (N_timesteps, N_assets, 768)
- data/processed/news_embeddings_metadata.json: Metadata (timestamps, asset_list)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.news import NewsFetcher
from features.embeddings import FinBERTEmbedder


def aggregate_headlines_by_time(
    df: pd.DataFrame,
    headlines_dict: Dict[str, List[Dict]],
    window_hours: int = 24
) -> pd.DataFrame:
    """
    Aggregate headlines by time window and asset.
    
    For each (timestamp, ticker) pair, collect all headlines within the
    time window and create a single aggregated text representation.
    
    Args:
        df: DataFrame with columns: timestamp, ticker
        headlines_dict: Dict mapping ticker -> List of headline dicts with 'timestamp', 'title'
        window_hours: Time window in hours for aggregating headlines (default: 24)
    
    Returns:
        pd.DataFrame: DataFrame with columns: timestamp, ticker, aggregated_text
    """
    print(f"  Aggregating headlines by time window ({window_hours} hours)...")
    
    # Create a copy of the dataframe
    result_df = df[['timestamp', 'ticker']].copy()
    result_df['aggregated_text'] = ""
    
    # Group by ticker and timestamp
    for (ticker, timestamp), group in result_df.groupby(['ticker', 'timestamp']):
        # Get headlines for this ticker
        ticker_headlines = headlines_dict.get(ticker, [])
        
        if not ticker_headlines:
            continue
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(pd.Series([timestamp])):
            timestamp = pd.to_datetime(timestamp)
        
        # Filter headlines within time window
        window_start = timestamp - pd.Timedelta(hours=window_hours)
        window_end = timestamp
        
        relevant_headlines = []
        for headline in ticker_headlines:
            headline_time = pd.to_datetime(headline.get('timestamp', timestamp))
            if window_start <= headline_time <= window_end:
                relevant_headlines.append(headline.get('title', ''))
        
        # Aggregate: Join headlines with separator
        if relevant_headlines:
            aggregated = " | ".join(relevant_headlines)
            result_df.loc[(result_df['ticker'] == ticker) & 
                         (result_df['timestamp'] == timestamp), 'aggregated_text'] = aggregated
    
    return result_df


def main():
    """Main execution flow for generating embeddings."""
    print("=" * 80)
    print("🔤 FinBERT Embedding Generator")
    print("Research Track Sprint 1: Semantic Grounding with Embeddings")
    print("=" * 80)
    print()
    
    # Step 1: Load training data to get structure
    print("Step 1: Loading training data structure...")
    data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
    
    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            project_root / "data" / "processed" / "crypto_deep_master.parquet",
            project_root / "data" / "processed" / "crypto_deep_v3.parquet",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_path = alt_path
                break
    
    if not data_path.exists():
        print(f"✗ Error: Training data not found. Please run data pipeline first.")
        sys.exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded training data: {len(df):,} rows")
    print(f"  Assets: {df['ticker'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Step 2: Fetch news headlines for all assets
    print("Step 2: Fetching news headlines...")
    news_fetcher = NewsFetcher()
    unique_tickers = sorted(df['ticker'].unique())
    
    headlines_dict = {}
    for ticker in unique_tickers:
        print(f"  Fetching news for {ticker}...")
        try:
            # Fetch recent headlines (last 90 days for historical coverage)
            headlines = news_fetcher.get_headlines(ticker, limit=100, days_back=90)
            headlines_dict[ticker] = headlines
            print(f"    ✓ Found {len(headlines)} headlines")
        except Exception as e:
            print(f"    ⚠ Error fetching news for {ticker}: {e}")
            headlines_dict[ticker] = []
    
    print()
    
    # Step 3: Aggregate headlines by time window
    print("Step 3: Aggregating headlines by time window...")
    df_with_text = aggregate_headlines_by_time(df, headlines_dict, window_hours=24)
    
    # Count non-empty texts
    non_empty = (df_with_text['aggregated_text'] != '').sum()
    print(f"  ✓ {non_empty:,} rows have headlines ({non_empty/len(df_with_text)*100:.1f}%)")
    print()
    
    # Step 4: Generate embeddings
    print("Step 4: Generating FinBERT embeddings...")
    embedder = FinBERTEmbedder()
    
    # Get unique aggregated texts (to avoid redundant computation)
    unique_texts = df_with_text['aggregated_text'].unique()
    text_to_embedding = {}
    
    # Process unique texts
    print(f"  Processing {len(unique_texts):,} unique text samples...")
    batch_size = 32
    for i in range(0, len(unique_texts), batch_size):
        batch_texts = unique_texts[i:i + batch_size]
        batch_embeddings = embedder.generate_embeddings(batch_texts.tolist(), batch_size=batch_size)
        
        # Map text to embedding
        for text, embedding in zip(batch_texts, batch_embeddings):
            text_to_embedding[text] = embedding
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"    Processed {min(i + batch_size, len(unique_texts)):,} / {len(unique_texts):,} samples")
    
    print(f"  ✓ Generated embeddings for {len(text_to_embedding):,} unique texts")
    print()
    
    # Step 5: Map embeddings back to DataFrame structure
    print("Step 5: Mapping embeddings to DataFrame structure...")
    embeddings_list = []
    for _, row in df_with_text.iterrows():
        text = row['aggregated_text']
        if text and text in text_to_embedding:
            embeddings_list.append(text_to_embedding[text])
        else:
            # Zero vector for no news
            embeddings_list.append(np.zeros(768, dtype=np.float32))
    
    df_with_text['embedding'] = embeddings_list
    print(f"  ✓ Mapped embeddings to {len(df_with_text):,} rows")
    print()
    
    # Step 6: Reshape to (N_timesteps, N_assets, 768)
    print("Step 6: Reshaping embeddings to (timesteps, assets, 768)...")
    
    # Get unique timestamps and assets (sorted for consistency)
    timestamps = sorted(df_with_text['timestamp'].unique())
    assets = sorted(df_with_text['ticker'].unique())
    
    # Create 3D array: (timesteps, assets, embedding_dim)
    n_timesteps = len(timestamps)
    n_assets = len(assets)
    embedding_dim = 768
    
    embeddings_array = np.zeros((n_timesteps, n_assets, embedding_dim), dtype=np.float32)
    
    # Create mapping for fast lookup
    df_indexed = df_with_text.set_index(['timestamp', 'ticker'])
    
    for t_idx, timestamp in enumerate(timestamps):
        for a_idx, asset in enumerate(assets):
            if (timestamp, asset) in df_indexed.index:
                embedding = df_indexed.loc[(timestamp, asset), 'embedding']
                embeddings_array[t_idx, a_idx, :] = embedding
    
    print(f"  ✓ Reshaped to array shape: {embeddings_array.shape}")
    print()
    
    # Step 7: Save embeddings and metadata
    print("Step 7: Saving embeddings and metadata...")
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy array
    embeddings_path = output_dir / "news_embeddings.npy"
    np.save(str(embeddings_path), embeddings_array)
    print(f"  ✓ Saved embeddings to: {embeddings_path}")
    print(f"    File size: {embeddings_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    # Save metadata
    metadata = {
        'timestamps': [str(ts) for ts in timestamps],
        'assets': assets,
        'shape': list(embeddings_array.shape),
        'embedding_dim': embedding_dim,
        'model': 'ProsusAI/finbert'
    }
    
    metadata_path = output_dir / "news_embeddings_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to: {metadata_path}")
    print()
    
    # Step 8: Summary statistics
    print("Step 8: Summary statistics...")
    non_zero_embeddings = np.sum(np.any(embeddings_array != 0, axis=2))
    total_cells = embeddings_array.shape[0] * embeddings_array.shape[1]
    coverage = (non_zero_embeddings / total_cells) * 100
    
    print(f"  Embedding coverage: {coverage:.1f}% ({non_zero_embeddings:,} / {total_cells:,} cells)")
    print(f"  Mean embedding norm: {np.linalg.norm(embeddings_array, axis=2).mean():.4f}")
    print(f"  Std embedding norm: {np.linalg.norm(embeddings_array, axis=2).std():.4f}")
    print()
    
    print("=" * 80)
    print("✅ Embedding generation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Verify embeddings: Check data/processed/news_embeddings.npy")
    print("  2. Integrate into environment: Update FastCryptoEnv to use Dict observation space")
    print("  3. Create multimodal policy: Implement MultimodalExtractor for two-tower architecture")
    print()

if __name__ == "__main__":
    main()

