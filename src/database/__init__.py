"""
Database Module - Persistence Layer

This module provides SQLite persistence using SQLAlchemy 2.0 ORM.
Enables long-term memory for the trading bot, allowing audit trails
and performance tracking independent of the broker.
"""

from .base import Base, get_engine, get_session_factory
from .models import PortfolioSnapshot, Trade
from .service import DatabaseService

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "Trade",
    "PortfolioSnapshot",
    "DatabaseService",
]

