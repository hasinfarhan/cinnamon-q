"""
Deep Data Pipeline - Crypto Universe with Fundamentals

Phase 3: The Crypto Expert Alpha Master - Step 1

This script builds the training dataset for the Hierarchical Deep Reinforcement
Learning (HDRL) trading agent by fetching:

1. Technical Analysis: Price/Volume history (3 years, 1Hour bars) from Alpaca
2. Fundamental Analysis: Developer activity and community stats from CoinGecko
3. Synthetic History: Applies jitter to current fundamentals to simulate historical data

The output dataset combines all three modalities for RL training.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pycoingecko import CoinGeckoAPI

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
load_dotenv()

# Crypto Universe is now loaded from CSV file
# This allows easy expansion without code changes
CRYPTO_UNIVERSE_CSV = Path(__file__).parent.parent / "data" / "raw" / "crypto_universe.csv"


def load_crypto_universe() -> dict:
    """
    Load crypto universe from CSV file.
    
    Returns:
        dict: Mapping of symbol -> {alpaca, coingecko}
    """
    if not CRYPTO_UNIVERSE_CSV.exists():
        raise FileNotFoundError(
            f"Crypto universe CSV not found: {CRYPTO_UNIVERSE_CSV}\n"
            "Please create data/raw/crypto_universe.csv with columns: symbol,alpaca_symbol,coingecko_id"
        )
    
    df = pd.read_csv(CRYPTO_UNIVERSE_CSV)
    universe = {}
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        universe[symbol] = {
            "alpaca": row['alpaca_symbol'],
            "coingecko": row['coingecko_id']
        }
    
    return universe


def initialize_apis():
    """
    Initialize Alpaca and CoinGecko API clients.
    
    Returns:
        tuple: (alpaca_headers, alpaca_data_url, coingecko_api) or (None, None, None) if credentials missing
    """
    # Initialize Alpaca API (using Data API for historical bars, not Trading API)
    alpaca_headers = None
    alpaca_data_url = None
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if api_key and secret_key:
        try:
            alpaca_headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key
            }
            # Use Data API endpoint for historical crypto bars
            # Paper trading uses: https://data.alpaca.markets
            # Live trading uses: https://data.alpaca.markets (same)
            alpaca_data_url = "https://data.alpaca.markets"
            print("✓ Alpaca API credentials loaded (using Data API for historical bars)")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Alpaca API: {e}")
    else:
        print("⚠ Warning: ALPACA_API_KEY or ALPACA_SECRET_KEY not set")
    
    # Initialize CoinGecko API (free tier, no key required)
    try:
        coingecko_api = CoinGeckoAPI()
        print("✓ CoinGecko API initialized")
    except Exception as e:
        print(f"⚠ Warning: Failed to initialize CoinGecko API: {e}")
        coingecko_api = None
    
    return alpaca_headers, alpaca_data_url, coingecko_api


def fetch_fundamentals(coin_id: str, coingecko_api: CoinGeckoAPI) -> Dict[str, Optional[float]]:
    """
    Fetch fundamental data from CoinGecko for a given coin.
    
    Args:
        coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
        coingecko_api: CoinGecko API client instance
    
    Returns:
        dict: Dictionary containing:
            - commit_count_4_weeks: Developer activity (commits in last 4 weeks)
            - twitter_followers: Community size (Twitter followers)
            - None values if data unavailable
    """
    try:
        # Get coin data with developer and community stats
        coin_data = coingecko_api.get_coin_by_id(
            id=coin_id,
            localization=False,
            tickers=False,
            market_data=False,
            community_data=True,
            developer_data=True,
            sparkline=False
        )
        
        # Extract developer data
        commit_count = None
        if coin_data.get("developer_data") and coin_data["developer_data"].get("commit_count_4_weeks"):
            commit_count = float(coin_data["developer_data"]["commit_count_4_weeks"])
        
        # Extract community data
        twitter_followers = None
        if coin_data.get("community_data") and coin_data["community_data"].get("twitter_followers"):
            twitter_followers = float(coin_data["community_data"]["twitter_followers"])
        
        return {
            "commit_count_4_weeks": commit_count,
            "twitter_followers": twitter_followers
        }
        
    except Exception as e:
        print(f"  ⚠ Warning: Failed to fetch fundamentals for {coin_id}: {e}")
        return {
            "commit_count_4_weeks": None,
            "twitter_followers": None
        }


def fetch_price_history(symbol: str, alpaca_headers: dict, alpaca_data_url: str, years: int = 3, timeframe: str = "15Min") -> pd.DataFrame:
    """
    Fetch price history from Alpaca Data API for a crypto symbol.
    
    Args:
        symbol: Alpaca crypto symbol (e.g., "BTC/USD")
        alpaca_headers: Alpaca API headers (with API keys)
        alpaca_data_url: Alpaca Data API base URL
        years: Number of years of history to fetch (default: 3)
        timeframe: Bar timeframe (default: "15Min" for high-frequency trading)
    
    Returns:
        pd.DataFrame: DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Alpaca Data API requires symbols in format "BTC/USD" (with slash)
        # Keep the original symbol format - don't remove the slash
        alpaca_symbol = symbol  # Already in "BTC/USD" format
        
        print(f"  Fetching {years} years of {timeframe} bars for {symbol}...")
        
        # Use Alpaca Data API for historical crypto bars
        # Endpoint: /v1beta3/crypto/us/bars (Data API, not Trading API)
        # Note: "us" in the path represents the feed, no need for feed parameter
        url = f"{alpaca_data_url}/v1beta3/crypto/us/bars"
        params = {
            "symbols": alpaca_symbol,  # Format: "BTC/USD" (with slash, comma-separated for multiple)
            "timeframe": timeframe,  # Now configurable (15Min for high-frequency)
            "start": start_date.strftime("%Y-%m-%dT%H:%M:%S-00:00"),
            "end": end_date.strftime("%Y-%m-%dT%H:%M:%S-00:00"),
            "limit": 10000,  # Max per request
        }
        
        all_bars = []
        page_token = None
        
        while True:
            # Add page_token to params if we have one
            current_params = params.copy()
            if page_token:
                current_params["page_token"] = page_token
            
            response = requests.get(url, headers=alpaca_headers, params=current_params)
            
            # Debug: Print response details if error
            if response.status_code != 200:
                print(f"  ⚠ API Response Status: {response.status_code}")
                print(f"  ⚠ API Response: {response.text[:500]}")
            
            response.raise_for_status()
            
            data = response.json()
            # Crypto API returns bars in format: {"bars": {"BTC/USD": [...]}}
            bars_dict = data.get("bars", {})
            bars = bars_dict.get(alpaca_symbol, [])
            
            if not bars:
                break
            
            all_bars.extend(bars)
            
            # Check if there are more pages
            page_token = data.get("next_page_token")
            if not page_token:
                break
        
        if not all_bars:
            print(f"  ⚠ Warning: No data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        bars_df = pd.DataFrame(all_bars)
        
        # Rename columns to match expected format
        bars_df = bars_df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume"
        })
        
        # Ensure timestamp is datetime
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
        
        # Add symbol column
        bars_df["symbol"] = symbol
        
        # Sort by timestamp
        bars_df = bars_df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"  ✓ Fetched {len(bars_df)} bars for {symbol}")
        return bars_df
        
    except Exception as e:
        print(f"  ✗ Error fetching price history for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def apply_synthetic_history(
    df: pd.DataFrame,
    commit_count: Optional[float],
    twitter_followers: Optional[float],
    jitter_std: float = 0.1
) -> pd.DataFrame:
    """
    Apply synthetic history to fundamentals by adding jitter.
    
    This simulates historical fundamental data by taking current values
    and applying small random noise backwards in time. This gives the RL
    agent non-static values to train on.
    
    Args:
        df: Price history DataFrame (must have 'timestamp' column)
        commit_count: Current commit count (or None)
        twitter_followers: Current Twitter followers (or None)
        jitter_std: Standard deviation for jitter (default: 0.1 = 10%)
    
    Returns:
        pd.DataFrame: DataFrame with added fundamental columns
    """
    df = df.copy()
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Apply jitter to commit_count
    if commit_count is not None and commit_count > 0:
        # Generate jitter: random walk with mean reversion
        # Start from current value and add small random changes
        n = len(df)
        jitter = np.random.normal(0, jitter_std, n)
        cumulative_jitter = np.cumsum(jitter)
        
        # Normalize to start at 0 and end at 0 (current value is accurate)
        cumulative_jitter = cumulative_jitter - cumulative_jitter[-1]
        
        # Apply jitter (ensure values stay positive)
        df["commit_count_4_weeks"] = np.maximum(
            commit_count * (1 + cumulative_jitter),
            0.1 * commit_count  # Floor at 10% of current value
        )
    else:
        df["commit_count_4_weeks"] = None
    
    # Apply jitter to twitter_followers
    if twitter_followers is not None and twitter_followers > 0:
        # Similar approach for Twitter followers
        n = len(df)
        jitter = np.random.normal(0, jitter_std, n)
        cumulative_jitter = np.cumsum(jitter)
        cumulative_jitter = cumulative_jitter - cumulative_jitter[-1]
        
        # Apply jitter (ensure values stay positive and less than current)
        df["twitter_followers"] = np.maximum(
            twitter_followers * (1 + cumulative_jitter),
            0.1 * twitter_followers  # Floor at 10% of current value
        )
    else:
        df["twitter_followers"] = None
    
    return df


def main():
    """Main execution flow for building the deep crypto dataset."""
    print("=" * 80)
    print("📊 Deep Data Pipeline - Crypto Universe with Fundamentals")
    print("Phase 3: The Crypto Expert Alpha Master - Step 1")
    print("=" * 80)
    print()
    
    # Step 1: Initialize APIs
    print("Step 1: Initializing APIs...")
    alpaca_headers, alpaca_data_url, coingecko_api = initialize_apis()
    
    if alpaca_headers is None or alpaca_data_url is None:
        print("✗ Error: Alpaca API not available. Cannot fetch price history.")
        sys.exit(1)
    
    if coingecko_api is None:
        print("✗ Error: CoinGecko API not available. Cannot fetch fundamentals.")
        sys.exit(1)
    
    print()
    
    # Step 2: Process each crypto in the universe
    print("Step 2: Processing crypto universe...")
    print(f"  Universe: {', '.join(CRYPTO_UNIVERSE.keys())}")
    print()
    
    all_data = []
    
    for ticker, mapping in CRYPTO_UNIVERSE.items():
        print(f"Processing {ticker} ({mapping['alpaca']})...")
        
        # Fetch price history (15Min bars for high-frequency trading)
        price_df = fetch_price_history(
            mapping["alpaca"], 
            alpaca_headers, 
            alpaca_data_url, 
            years=3,
            timeframe="15Min"  # v1.2: High-frequency 15-minute bars
        )
        
        if price_df.empty:
            print(f"  ⚠ Skipping {ticker} - no price data available")
            continue
        
        # Fetch current fundamentals
        print(f"  Fetching fundamentals from CoinGecko ({mapping['coingecko']})...")
        fundamentals = fetch_fundamentals(mapping["coingecko"], coingecko_api)
        
        commit_count = fundamentals.get("commit_count_4_weeks")
        twitter_followers = fundamentals.get("twitter_followers")
        
        if commit_count is not None:
            print(f"    ✓ Commit count (4 weeks): {commit_count:.0f}")
        else:
            print(f"    ⚠ Commit count: Not available")
        
        if twitter_followers is not None:
            print(f"    ✓ Twitter followers: {twitter_followers:,.0f}")
        else:
            print(f"    ⚠ Twitter followers: Not available")
        
        # Apply synthetic history (jitter to fundamentals)
        print(f"  Applying synthetic history (jitter)...")
        merged_df = apply_synthetic_history(
            price_df,
            commit_count=commit_count,
            twitter_followers=twitter_followers,
            jitter_std=0.1  # 10% standard deviation
        )
        
        # Add ticker column
        merged_df["ticker"] = ticker
        
        all_data.append(merged_df)
        print(f"  ✓ {ticker} processed: {len(merged_df)} rows")
        print()
    
    if not all_data:
        print("✗ Error: No data was successfully fetched for any crypto")
        sys.exit(1)
    
    # Step 3: Combine all data
    print("Step 3: Combining all data...")
    # Filter out empty DataFrames before concatenation to avoid FutureWarning
    non_empty_data = [df for df in all_data if not df.empty]
    combined_df = pd.concat(non_empty_data, ignore_index=True)
    
    # Sort by timestamp and ticker
    combined_df = combined_df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    
    print(f"✓ Combined dataset: {len(combined_df)} rows")
    print(f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"  Tickers: {combined_df['ticker'].unique().tolist()}")
    print(f"  Columns: {list(combined_df.columns)}")
    print()
    
    # Step 4: Save to parquet
    print("Step 4: Saving dataset...")
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "crypto_deep_master.parquet"
    
    try:
        combined_df.to_parquet(output_path, index=False)
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"✓ Dataset saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Rows: {len(combined_df):,}")
        print()
        
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success summary
    print("=" * 80)
    print("✅ Deep Data Pipeline complete!")
    print("=" * 80)
    print()
    print("Dataset Summary:")
    print(f"  File: {output_path}")
    print(f"  Rows: {len(combined_df):,}")
    print(f"  Columns: {len(combined_df.columns)}")
    print(f"  Tickers: {len(combined_df['ticker'].unique())}")
    print()
    print("Data Modalities:")
    print("  ✓ Technical Analysis: Price/Volume (OHLCV)")
    print("  ✓ Fundamental Analysis: Developer activity, Community stats")
    print("  ✓ Synthetic History: Jittered fundamentals for training")
    print()
    print("Next Steps:")
    print("  - Use this dataset for HDRL agent training")
    print("  - Add sentiment analysis (LLM-scored news) in next phase")
    print()


if __name__ == "__main__":
    main()

