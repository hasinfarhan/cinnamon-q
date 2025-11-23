"""
Market Data Fetcher - Module 1

This script downloads historical market data for the MVP universe of stocks
and saves it in a format ready for the Strategy Engine.

Data Format:
- Columns: ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
- Period: 5 years of historical data
- Format: Parquet (efficient for pandas)
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add src to the Python path (for future use)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    """Fetch and save market data for the MVP universe."""
    print("=" * 80)
    print("📊 Market Data Fetcher - Module 1")
    print("=" * 80)
    print()
    
    # Step 1: Define MVP Universe
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    print(f"Step 1: Fetching data for {len(TICKERS)} tickers...")
    print(f"  Universe: {', '.join(TICKERS)}")
    print()
    
    # Step 2: Download data using yfinance
    print("Step 2: Downloading 5-year historical data from Yahoo Finance...")
    try:
        # Download data with 5-year period
        # group_by='ticker' returns MultiIndex DataFrame
        raw_data = yf.download(
            TICKERS,
            period="5y",
            group_by='ticker',
            progress=True
        )
        print(f"✓ Downloaded data: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
        print()
    except Exception as e:
        print(f"✗ Error downloading data: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Flatten the MultiIndex columns
    print("Step 3: Processing and flattening data...")
    try:
        # The raw data has MultiIndex columns: (Ticker, OHLCV)
        # We need to flatten this to: Date, Ticker, Open, High, Low, Close, Volume
        
        processed_rows = []
        
        # Iterate through each ticker
        for ticker in TICKERS:
            if ticker not in raw_data.columns.levels[0]:
                print(f"  ⚠ Warning: {ticker} not found in downloaded data, skipping...")
                continue
            
            # Extract data for this ticker
            ticker_data = raw_data[ticker].copy()
            
            # Reset index to get Date as a column
            ticker_data = ticker_data.reset_index()
            
            # Add ticker column
            ticker_data['ticker'] = ticker
            
            # Select and rename columns to match expected format
            ticker_data = ticker_data[[
                'Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume'
            ]]
            
            # Rename Date to date (lowercase for consistency)
            ticker_data = ticker_data.rename(columns={'Date': 'date'})
            
            # Convert column names to lowercase
            ticker_data.columns = ticker_data.columns.str.lower()
            
            # Append to processed rows
            processed_rows.append(ticker_data)
        
        # Combine all tickers into single DataFrame
        if not processed_rows:
            raise ValueError("No data was successfully downloaded for any ticker")
        
        processed_data = pd.concat(processed_rows, ignore_index=True)
        
        # Sort by date and ticker
        processed_data = processed_data.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        print(f"✓ Processed data: {len(processed_data)} rows")
        print(f"  Date range: {processed_data['date'].min()} to {processed_data['date'].max()}")
        print(f"  Tickers: {processed_data['ticker'].unique().tolist()}")
        print()
        
    except Exception as e:
        print(f"✗ Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Save to parquet
    print("Step 4: Saving data to parquet file...")
    try:
        # Ensure directory exists
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "market_data.parquet"
        
        # Save to parquet
        processed_data.to_parquet(output_path, index=False)
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"✓ Data saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Rows: {len(processed_data):,}")
        print(f"  Columns: {list(processed_data.columns)}")
        print()
        
    except Exception as e:
        print(f"✗ Error saving data: {str(e)}")
        print()
        print("Note: You may need to install pyarrow or fastparquet:")
        print("  pipenv install pyarrow")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success summary
    print("=" * 80)
    print("✅ Market data fetch completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Verify data: Check data/processed/market_data.parquet")
    print("  2. Run bot: python scripts/run_bot.py")
    print()


if __name__ == "__main__":
    main()

