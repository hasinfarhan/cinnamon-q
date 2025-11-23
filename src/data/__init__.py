"""
Data Module - Data Sources and Providers

This module provides data fetching capabilities for various sources:
- News providers (yfinance, NewsAPI, etc.)
- Market data providers
- Fundamental data providers
"""

from .news import NewsFetcher

__all__ = ["NewsFetcher"]
