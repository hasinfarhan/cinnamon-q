#!/usr/bin/env python3
"""
Data Ingestion Script for BTC-USD Quantitative Analysis

This script downloads BTC-USD daily data from 2020-01-01 to 2025-08-31,
calculates RSI and MACD technical indicators using pandas-ta,
and saves the data to data/agent_a_data.csv.

Author: Cinnamon-Q Project
Date: 2024
"""

import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_btc_data(start_date='2020-01-01', end_date='2025-08-31'):
    """
    Download BTC-USD daily data using yfinance.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: BTC-USD daily data
    """
    logger.info(f"Downloading BTC-USD data from {start_date} to {end_date}")
    
    try:
        # Download BTC-USD data
        btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        
        if btc.empty:
            raise ValueError("No data downloaded. Check date range and ticker symbol.")
        
        logger.info(f"Successfully downloaded {len(btc)} days of BTC-USD data")
        return btc
        
    except Exception as e:
        logger.error(f"Error downloading BTC-USD data: {e}")
        raise

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices (pd.Series): Price series
        period (int): RSI period (default 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices (pd.Series): Price series
        fast (int): Fast EMA period (default 12)
        slow (int): Slow EMA period (default 26)
        signal (int): Signal line EMA period (default 9)
    
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def add_technical_indicators(df):
    """
    Add RSI and MACD technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): Price data with OHLCV columns
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    logger.info("Adding technical indicators (RSI and MACD)")
    
    try:
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Add RSI (Relative Strength Index) with default 14-period
        data['RSI'] = calculate_rsi(data['Close'], period=14)
        
        # Add MACD (Moving Average Convergence Divergence) with default parameters
        macd_line, signal_line, histogram = calculate_macd(data['Close'], fast=12, slow=26, signal=9)
        data['MACD'] = macd_line
        data['MACD_Signal'] = signal_line
        data['MACD_Histogram'] = histogram
        
        logger.info("Successfully added RSI and MACD indicators")
        return data
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        raise

def save_data_to_csv(df, filepath='data/agent_a_data.csv'):
    """
    Save the DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save the CSV file
    """
    logger.info(f"Saving data to {filepath}")
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to CSV
        df.to_csv(filepath)
        
        logger.info(f"Successfully saved {len(df)} rows to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        raise

def main():
    """
    Main function to orchestrate the data ingestion process.
    """
    logger.info("Starting BTC-USD data ingestion process")
    
    try:
        # Step 1: Download BTC-USD data
        btc_data = download_btc_data()
        
        # Step 2: Add technical indicators
        btc_data_with_indicators = add_technical_indicators(btc_data)
        
        # Step 3: Save to CSV
        save_data_to_csv(btc_data_with_indicators)
        
        # Display summary information
        logger.info("Data ingestion completed successfully!")
        logger.info(f"Final dataset shape: {btc_data_with_indicators.shape}")
        logger.info(f"Columns: {list(btc_data_with_indicators.columns)}")
        logger.info(f"Date range: {btc_data_with_indicators.index.min()} to {btc_data_with_indicators.index.max()}")
        
        # Display first few rows
        print("\nFirst 5 rows of the dataset:")
        print(btc_data_with_indicators.head())
        
        # Display last few rows
        print("\nLast 5 rows of the dataset:")
        print(btc_data_with_indicators.tail())
        
        return btc_data_with_indicators
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
