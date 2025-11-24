"""
Data Ingestion Stage - Fetch Raw Market Data

This module wraps the data fetching logic from scripts/08_fetch_crypto_deep.py
into a callable function for the MLOps pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pycoingecko import CoinGeckoAPI

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Crypto Universe CSV path
CRYPTO_UNIVERSE_CSV = Path(__file__).parent.parent.parent / "data" / "raw" / "crypto_universe.csv"


def load_crypto_universe() -> dict:
    """Load crypto universe from CSV file."""
    if not CRYPTO_UNIVERSE_CSV.exists():
        raise FileNotFoundError(
            f"Crypto universe CSV not found: {CRYPTO_UNIVERSE_CSV}\n"
            "Please create data/raw/crypto_universe.csv"
        )
    
    df = pd.read_csv(CRYPTO_UNIVERSE_CSV)
    universe = {}
    for _, row in df.iterrows():
        universe[row['symbol']] = {
            'alpaca': row['alpaca_symbol'],
            'coingecko': row['coingecko_id']
        }
    return universe


def initialize_apis() -> tuple:
    """Initialize Alpaca and CoinGecko APIs."""
    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not alpaca_api_key or not alpaca_secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    
    alpaca_headers = {
        "APCA-API-KEY-ID": alpaca_api_key,
        "APCA-API-SECRET-KEY": alpaca_secret_key
    }
    alpaca_data_url = "https://data.alpaca.markets"
    
    coingecko_api = CoinGeckoAPI()
    
    return alpaca_headers, alpaca_data_url, coingecko_api


def fetch_fundamentals(coin_id: str, coingecko_api: CoinGeckoAPI) -> Dict[str, float]:
    """Fetch fundamental data from CoinGecko."""
    try:
        coin_data = coingecko_api.get_coin_by_id(coin_id, localization=False, tickers=False, market_data=False, community_data=True, developer_data=True)
        
        commit_count = coin_data.get('developer_data', {}).get('commit_count_4_weeks', 0) or 0
        twitter_followers = coin_data.get('community_data', {}).get('twitter_followers', 0) or 0
        
        return {
            'commit_count_4_weeks': float(commit_count),
            'twitter_followers': float(twitter_followers)
        }
    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {coin_id}: {e}")
        return {'commit_count_4_weeks': 100.0, 'twitter_followers': 10000.0}


def fetch_price_history(symbol: str, alpaca_headers: dict, alpaca_data_url: str, years: int = 3, timeframe: str = "15Min") -> pd.DataFrame:
    """Fetch price history from Alpaca Data API."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    crypto_symbol = symbol if '/' in symbol else f"{symbol}/USD"
    
    url = f"{alpaca_data_url}/v1beta3/crypto/us/bars"
    params = {
        "symbols": crypto_symbol,
        "timeframe": timeframe,
        "start": start_date.strftime("%Y-%m-%dT%H:%M:%S-00:00"),
        "end": end_date.strftime("%Y-%m-%dT%H:%M:%S-00:00"),
        "limit": 10000,
    }
    
    all_bars = []
    page_token = None
    
    while True:
        current_params = params.copy()
        if page_token:
            current_params["page_token"] = page_token
        
        response = requests.get(url, headers=alpaca_headers, params=current_params)
        response.raise_for_status()
        data = response.json()
        
        bars_dict = data.get("bars", {})
        bars = bars_dict.get(crypto_symbol, [])
        
        if not bars:
            break
        
        all_bars.extend(bars)
        
        page_token = data.get("next_page_token")
        if not page_token:
            break
    
    if not all_bars:
        return pd.DataFrame()
    
    bars_df = pd.DataFrame(all_bars)
    bars_df = bars_df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
    bars_df["ticker"] = crypto_symbol.split('/')[0]
    
    return bars_df.sort_values("timestamp").reset_index(drop=True)


def apply_synthetic_history(df: pd.DataFrame, commit_count: float, twitter_followers: float, jitter_std: float = 0.1) -> pd.DataFrame:
    """Apply synthetic historical fundamentals with jitter."""
    df = df.copy()
    n_rows = len(df)
    
    # Random walk with mean reversion
    current_commit = commit_count
    current_twitter = twitter_followers
    
    commits = []
    twitters = []
    
    for i in range(n_rows):
        # Add jitter (random walk)
        current_commit += np.random.normal(0, current_commit * jitter_std)
        current_twitter += np.random.normal(0, current_twitter * jitter_std)
        
        # Mean reversion (tendency to return to current value)
        current_commit = 0.99 * current_commit + 0.01 * commit_count
        current_twitter = 0.99 * current_twitter + 0.01 * twitter_followers
        
        # Ensure non-negative
        current_commit = max(0, current_commit)
        current_twitter = max(0, current_twitter)
        
        commits.append(current_commit)
        twitters.append(current_twitter)
    
    df['commit_count_4_weeks'] = commits
    df['twitter_followers'] = twitters
    
    return df


def run_ingestion(output_path: Optional[Path] = None, years: int = 3, timeframe: str = "15Min") -> Path:
    """
    Run the data ingestion stage.
    
    Fetches price history and fundamentals for all assets in the crypto universe,
    applies synthetic history, and saves to parquet.
    
    Args:
        output_path: Path to save the output parquet file (default: data/processed/crypto_deep_master.parquet)
        years: Number of years of history to fetch (default: 3)
        timeframe: Bar timeframe (default: "15Min")
    
    Returns:
        Path to the saved parquet file
    """
    logger.info("=" * 80)
    logger.info("Stage 1: Data Ingestion")
    logger.info("=" * 80)
    
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "data" / "processed" / "crypto_deep_master.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load universe
    logger.info("Loading crypto universe...")
    crypto_universe = load_crypto_universe()
    logger.info(f"  Universe: {', '.join(crypto_universe.keys())}")
    
    # Initialize APIs
    logger.info("Initializing APIs...")
    alpaca_headers, alpaca_data_url, coingecko_api = initialize_apis()
    
    # Fetch data for each asset
    all_dataframes = []
    
    for ticker, mapping in crypto_universe.items():
        logger.info(f"Processing {ticker}...")
        
        # Fetch fundamentals
        fundamentals = fetch_fundamentals(mapping['coingecko'], coingecko_api)
        logger.info(f"  Fundamentals: commits={fundamentals['commit_count_4_weeks']:.0f}, "
                   f"twitter={fundamentals['twitter_followers']:.0f}")
        
        # Fetch price history
        price_df = fetch_price_history(
            mapping['alpaca'],
            alpaca_headers,
            alpaca_data_url,
            years=years,
            timeframe=timeframe
        )
        
        if price_df.empty:
            logger.warning(f"  No price data for {ticker}")
            continue
        
        logger.info(f"  Fetched {len(price_df)} bars")
        
        # Apply synthetic history
        price_df = apply_synthetic_history(
            price_df,
            fundamentals['commit_count_4_weeks'],
            fundamentals['twitter_followers']
        )
        
        all_dataframes.append(price_df)
    
    # Combine all data
    if not all_dataframes:
        raise ValueError("No data fetched for any assets")
    
    combined_df = pd.concat([df for df in all_dataframes if not df.empty], ignore_index=True)
    combined_df = combined_df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
    
    # Save to parquet
    logger.info(f"Saving to {output_path}...")
    combined_df.to_parquet(output_path, index=False, compression='snappy')
    
    logger.info(f"✓ Ingestion complete: {len(combined_df):,} rows, "
               f"{combined_df['ticker'].nunique()} assets, "
               f"{output_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    return output_path

