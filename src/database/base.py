"""
Database Base Configuration

This module sets up SQLAlchemy 2.0 base classes and factory functions
for creating database engines and session factories.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    """
    Base class for all database models.
    
    This is the SQLAlchemy 2.0 DeclarativeBase that all ORM models
    should inherit from.
    """
    pass


def get_engine(url: str, echo: bool = False):
    """
    Create a SQLAlchemy engine for the given database URL.
    
    Args:
        url: Database connection URL (e.g., "sqlite:///arynosq.db")
        echo: If True, log all SQL statements (useful for debugging)
    
    Returns:
        Engine: SQLAlchemy engine instance
    """
    return create_engine(url, echo=echo)


def get_session_factory(engine):
    """
    Create a session factory for the given engine.
    
    Args:
        engine: SQLAlchemy engine instance
    
    Returns:
        sessionmaker: Session factory that can be used to create sessions
    """
    return sessionmaker(bind=engine, expire_on_commit=False)

