"""
News Provider - Fetch Crypto News Headlines

This module provides functionality to fetch news headlines for crypto assets
from various sources:
- CryptoPanic API (crypto-specific, recommended)
- yfinance (general market news, fallback)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NewsFetcher:
    """
    News fetcher for crypto assets.
    
    This class provides methods to fetch news headlines from multiple sources:
    - CryptoPanic API (crypto-specific, recommended) - https://cryptopanic.com/developers/api/
    - yfinance (general market news, fallback)
    
    CryptoPanic provides better crypto-specific news compared to yfinance's
    general market coverage.
    """
    
    def __init__(self, use_cryptopanic: bool = True, use_yfinance: bool = True):
        """
        Initialize the news fetcher.
        
        Args:
            use_cryptopanic: Whether to use CryptoPanic API (default: True)
            use_yfinance: Whether to use yfinance as fallback (default: True)
        """
        self.use_cryptopanic = use_cryptopanic
        self.use_yfinance = use_yfinance
        self.cryptopanic_api_key = os.getenv("CRYPTOPANIC_API_KEY", None)
        
        # CryptoPanic API base URL
        self.cryptopanic_base_url = "https://cryptopanic.com/api/v1"
    
    def _fetch_cryptopanic_news(
        self,
        symbol: str,
        limit: int = 50,
        days_back: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch news from CryptoPanic API.
        
        CryptoPanic is crypto-specific and provides better quality headlines
        for cryptocurrency assets.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH", "BTC-USD")
            limit: Maximum number of headlines
            days_back: Optional filter for recent news
        
        Returns:
            List of headline dictionaries
        """
        try:
            # CryptoPanic uses coin symbols without suffix (BTC, not BTC-USD)
            # Extract base symbol
            coin_symbol = symbol.replace("-USD", "").replace("/USD", "").upper()
            
            # Build API URL
            # Endpoint: /posts/?currencies=BTC&public=true
            url = f"{self.cryptopanic_base_url}/posts/"
            params = {
                "currencies": coin_symbol,
                "public": "true",  # Public API (no auth required for free tier)
                "kind": "news",  # Only news, not media
            }
            
            # Add API key if available (for higher rate limits)
            if self.cryptopanic_api_key:
                params["auth_token"] = self.cryptopanic_api_key
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return []
            
            # Process news items
            headlines = []
            cutoff_date = None
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in results[:limit]:
                # Extract timestamp
                created_at = item.get("created_at", "")
                if not created_at:
                    continue
                
                try:
                    # CryptoPanic returns ISO format: "2025-11-23T12:37:46Z"
                    if created_at.endswith("Z"):
                        created_at = created_at.replace("Z", "+00:00")
                    timestamp = datetime.fromisoformat(created_at)
                except (ValueError, AttributeError):
                    # Try alternative parsing
                    try:
                        from dateutil import parser
                        timestamp = parser.parse(created_at)
                    except:
                        continue
                
                # Filter by date if specified
                if cutoff_date:
                    # Handle timezone-aware comparison
                    if timestamp.tzinfo is not None and cutoff_date.tzinfo is None:
                        from datetime import timezone
                        cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
                    elif timestamp.tzinfo is None and cutoff_date.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)
                        cutoff_date = cutoff_date.replace(tzinfo=None)
                    
                    if timestamp < cutoff_date:
                        continue
                
                # Extract title and source
                title = item.get("title", "")
                source = item.get("source", {}).get("title", "CryptoPanic")
                
                if title:
                    headlines.append({
                        "timestamp": timestamp,
                        "title": title,
                        "source": source
                    })
            
            return headlines
            
        except requests.exceptions.RequestException as e:
            # Silently fail - will fallback to yfinance
            return []
        except Exception as e:
            # Silently fail
            return []
    
    def _fetch_yfinance_news(
        self,
        symbol: str,
        limit: int = 50,
        days_back: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch news from yfinance (fallback source).
        
        Args:
            symbol: Crypto symbol (e.g., "BTC-USD", "ETH-USD", "BTC")
            limit: Maximum number of headlines
            days_back: Optional filter for recent news
        
        Returns:
            List of headline dictionaries
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
                # Use timezone-aware datetime for comparison
                cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=days_back)
            
            for item in news_list[:limit]:
                # Extract timestamp - yfinance uses multiple formats
                timestamp = None
                
                # Try providerPublishTime (Unix timestamp) first
                provider_time = item.get('providerPublishTime', 0)
                if provider_time:
                    try:
                        timestamp = datetime.fromtimestamp(provider_time)
                    except (ValueError, OSError):
                        pass
                
                # Fallback to pubDate (ISO string) - this is what yfinance actually uses
                if not timestamp:
                    pub_date = item.get('pubDate', '')
                    if pub_date:
                        try:
                            # Parse ISO format: '2025-11-23T12:37:46Z'
                            timestamp = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            # Try alternative parsing
                            try:
                                from dateutil import parser
                                timestamp = parser.parse(pub_date)
                            except:
                                pass
                
                # If still no timestamp, try content.pubDate
                if not timestamp:
                    content = item.get('content', {})
                    if isinstance(content, dict):
                        pub_date = content.get('pubDate', '')
                        if pub_date:
                            try:
                                timestamp = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            except (ValueError, AttributeError):
                                pass
                
                # Skip if we still don't have a timestamp
                if not timestamp:
                    continue
                
                # Filter by date if specified
                # Handle timezone-aware vs timezone-naive comparison
                if cutoff_date:
                    # Make cutoff_date timezone-aware if timestamp is timezone-aware
                    if timestamp.tzinfo is not None and cutoff_date.tzinfo is None:
                        # Make cutoff_date timezone-aware (use UTC)
                        from datetime import timezone
                        cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
                    elif timestamp.tzinfo is None and cutoff_date.tzinfo is not None:
                        # Make timestamp timezone-naive for comparison
                        timestamp = timestamp.replace(tzinfo=None)
                        cutoff_date = cutoff_date.replace(tzinfo=None)
                    
                    if timestamp < cutoff_date:
                        continue
                
                # Extract title - try multiple locations
                title = item.get('title', '')
                if not title:
                    content = item.get('content', {})
                    if isinstance(content, dict):
                        title = content.get('title', '')
                
                # Extract source - try multiple locations
                source = item.get('publisher', 'Unknown')
                if source == 'Unknown':
                    provider = item.get('provider', {})
                    if isinstance(provider, dict):
                        source = provider.get('displayName', 'Unknown')
                
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
    
    def get_headlines(
        self,
        symbol: str,
        limit: int = 50,
        days_back: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch news headlines for a crypto symbol from multiple sources.
        
        Tries sources in order:
        1. CryptoPanic API (crypto-specific, recommended)
        2. yfinance (general market news, fallback)
        
        Args:
            symbol: Crypto symbol (e.g., "BTC-USD", "ETH-USD", "BTC", "BTC/USD")
            limit: Maximum number of headlines to return (default: 50)
            days_back: Optional filter for news within N days (default: None = all)
        
        Returns:
            List of dictionaries with keys: 'timestamp', 'title', 'source'
        
        Note:
            CryptoPanic provides better crypto-specific news. yfinance is used
            as a fallback for general market coverage.
        """
        all_headlines = []
        
        # Try CryptoPanic first (crypto-specific)
        if self.use_cryptopanic:
            cryptopanic_news = self._fetch_cryptopanic_news(symbol, limit=limit, days_back=days_back)
            all_headlines.extend(cryptopanic_news)
        
        # If we have enough headlines, return them
        if len(all_headlines) >= limit:
            return all_headlines[:limit]
        
        # Fallback to yfinance if needed
        if self.use_yfinance and len(all_headlines) < limit:
            remaining_limit = limit - len(all_headlines)
            yfinance_news = self._fetch_yfinance_news(symbol, limit=remaining_limit, days_back=days_back)
            all_headlines.extend(yfinance_news)
        
        # Remove duplicates (same title) and sort by timestamp
        seen_titles = set()
        unique_headlines = []
        for headline in all_headlines:
            title_lower = headline["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_headlines.append(headline)
        
        # Sort by timestamp (newest first)
        unique_headlines.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return unique_headlines[:limit]
    
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

