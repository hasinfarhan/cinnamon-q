#!/usr/bin/env python3
"""
Data Verification Script

This script verifies that the generated CSV file contains the expected data
with proper RSI and MACD columns.
"""

import pandas as pd
import os

def verify_data():
    """Verify the generated CSV data."""
    csv_path = 'data/agent_a_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        return False
    
    try:
        # Load the data
        df = pd.read_csv(csv_path, index_col=0, header=[0,1])
        
        print("✅ CSV file loaded successfully")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required technical indicator columns present")
        
        # Check for data quality
        print("\n📈 Technical Indicators Summary:")
        for col in required_columns:
            if col in df.columns:
                non_null_count = df[col].count()
                print(f"  {col}: {non_null_count} non-null values")
        
        # Show sample data
        print("\n🔍 Sample data (first 3 rows):")
        print(df[['Close', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']].head(3))
        
        print("\n✅ Data verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        return False

if __name__ == "__main__":
    verify_data()
