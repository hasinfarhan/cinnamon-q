"""
News Provider - Fetch Crypto News Headlines

This module provides functionality to fetch news headlines for crypto assets
from various free sources (yfinance, etc.).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


class NewsFetcher:
    """
    News fetcher for crypto assets.
    
    This class provides methods to fetch news headlines from free sources
    like yfinance. For MVP, we use yfinance which provides recent news.
    """
    
    def __init__(self):
        """Initialize the news fetcher."""
        pass
    
    def get_headlines(
        self,
        symbol: str,
        limit: int = 50,
        days_back: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch news headlines for a crypto symbol.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC-USD", "ETH-USD", "BTC/USD")
                Note: yfinance uses format like "BTC-USD" for crypto
            limit: Maximum number of headlines to return (default: 50)
            days_back: Optional filter for news within N days (default: None = all)
        
        Returns:
            List of dictionaries with keys: 'timestamp', 'title', 'source'
        
        Note:
            yfinance news availability for crypto is limited. If no news is found,
            this method returns an empty list. Consider using alternative news
            sources (NewsAPI, CryptoCompare, etc.) for production.
        """
        try:
            # yfinance uses format like "BTC-USD" for crypto
            # If symbol is "BTC/USD", convert to "BTC-USD"
            yf_symbol = symbol.replace("/", "-")
            
            # Try alternative formats if the first one fails
            symbol_variants = [yf_symbol]
            if not yf_symbol.endswith("-USD"):
                symbol_variants.append(f"{yf_symbol}-USD")
            
            news_list = None
            for variant in symbol_variants:
                try:
                    ticker = yf.Ticker(variant)
                    news_list = ticker.news
                    if news_list:
                        break
                except:
                    continue
            
            if not news_list:
                return []
            
            # Process news items
            headlines = []
            cutoff_date = None
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in news_list[:limit]:
                # Extract timestamp
                provider_time = item.get('providerPublishTime', 0)
                if not provider_time:
                    continue
                    
                timestamp = datetime.fromtimestamp(provider_time)
                
                # Filter by date if specified
                if cutoff_date and timestamp < cutoff_date:
                    continue
                
                # Extract title and source
                title = item.get('title', '')
                source = item.get('publisher', 'Unknown')
                
                if title:
                    headlines.append({
                        'timestamp': timestamp,
                        'title': title,
                        'source': source
                    })
            
            return headlines
            
        except Exception as e:
            # Silently return empty list - news fetching is optional
            return []
    
    def get_headlines_dataframe(
        self,
        symbol: str,
        limit: int = 50,
        days_back: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch news headlines and return as DataFrame.
        
        Args:
            symbol: Crypto symbol
            limit: Maximum number of headlines
            days_back: Optional filter for recent news
        
        Returns:
            DataFrame with columns: timestamp, title, source
        """
        headlines = self.get_headlines(symbol, limit=limit, days_back=days_back)
        
        if not headlines:
            return pd.DataFrame(columns=['timestamp', 'title', 'source'])
        
        df = pd.DataFrame(headlines)
        df['symbol'] = symbol
        return df

