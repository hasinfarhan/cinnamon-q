"""
Database Service - High-Level Database Operations

This module provides the DatabaseService class that wraps database operations
for logging trades and portfolio snapshots.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session

from .base import Base, get_engine, get_session_factory
from .models import PortfolioSnapshot, Trade


class DatabaseService:
    """
    High-level service for database operations.
    
    This class provides methods to log trades and portfolio snapshots,
    abstracting away the SQLAlchemy session management.
    
    Attributes:
        engine: SQLAlchemy engine instance
        session_factory: Session factory for creating database sessions
    """
    
    def __init__(self, connection_url: str, echo: bool = False):
        """
        Initialize the database service.
        
        Args:
            connection_url: Database connection URL (e.g., "sqlite:///arynosq.db")
            echo: If True, log all SQL statements (useful for debugging)
        """
        self.engine = get_engine(connection_url, echo=echo)
        self.session_factory = get_session_factory(self.engine)
        
        # Create all tables if they don't exist
        Base.metadata.create_all(self.engine)
    
    def log_trade(
        self,
        strategy_name: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        notional: float,
        client_order_id: str,
        timestamp: Optional[datetime] = None
    ) -> Trade:
        """
        Log a trade execution to the database.
        
        This method creates a Trade record with all execution details.
        The client_order_id must be unique (enforced by database constraint).
        
        Args:
            strategy_name: Name of the strategy that generated this trade
            symbol: Trading symbol (e.g., "AAPL", "BTC/USD")
            side: Order side ("buy" or "sell")
            qty: Quantity of shares/coins traded
            price: Average fill price per unit
            notional: Total USD value of the trade
            client_order_id: Unique identifier for idempotency
            timestamp: UTC timestamp (defaults to now if not provided)
        
        Returns:
            Trade: The created Trade record
        
        Raises:
            IntegrityError: If client_order_id already exists (duplicate trade)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        trade = Trade(
            strategy_name=strategy_name,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            notional=notional,
            client_order_id=client_order_id,
            timestamp=timestamp
        )
        
        with self.session_factory() as session:
            session.add(trade)
            session.commit()
            session.refresh(trade)
        
        # SQLite doesn't preserve timezone info, so ensure timestamp is UTC-aware
        # when returned (SQLite returns naive datetime, we assume UTC)
        if trade.timestamp.tzinfo is None:
            trade.timestamp = trade.timestamp.replace(tzinfo=timezone.utc)
        
        return trade
    
    def log_trade_from_dict(self, trade_data: Dict) -> Trade:
        """
        Log a trade from a dictionary (convenience method).
        
        This method extracts trade information from a dictionary,
        typically from broker API responses.
        
        Args:
            trade_data: Dictionary containing trade information.
                Expected keys: strategy_name, symbol, side, qty, price,
                notional, client_order_id, timestamp (optional)
        
        Returns:
            Trade: The created Trade record
        """
        return self.log_trade(
            strategy_name=trade_data.get("strategy_name", "unknown"),
            symbol=trade_data.get("symbol"),
            side=trade_data.get("side"),
            qty=trade_data.get("qty", 0.0),
            price=trade_data.get("price", 0.0),
            notional=trade_data.get("notional", 0.0),
            client_order_id=trade_data.get("client_order_id"),
            timestamp=trade_data.get("timestamp")
        )
    
    def log_snapshot(
        self,
        total_equity: float,
        cash: float,
        timestamp: Optional[datetime] = None
    ) -> PortfolioSnapshot:
        """
        Log a portfolio snapshot to the database.
        
        This method creates a PortfolioSnapshot record with current
        portfolio state. Used for performance tracking over time.
        
        Args:
            total_equity: Total portfolio value in USD
            cash: Available cash balance in USD
            timestamp: UTC timestamp (defaults to now if not provided)
        
        Returns:
            PortfolioSnapshot: The created PortfolioSnapshot record
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        snapshot = PortfolioSnapshot(
            total_equity=total_equity,
            cash=cash,
            timestamp=timestamp
        )
        
        with self.session_factory() as session:
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
        
        # SQLite doesn't preserve timezone info, so ensure timestamp is UTC-aware
        # when returned (SQLite returns naive datetime, we assume UTC)
        if snapshot.timestamp.tzinfo is None:
            snapshot.timestamp = snapshot.timestamp.replace(tzinfo=timezone.utc)
        
        return snapshot
    
    def log_snapshot_from_state(self, state) -> PortfolioSnapshot:
        """
        Log a portfolio snapshot from a PortfolioState Pydantic model.
        
        This is a convenience method that extracts data from the
        PortfolioState model used by the execution engine.
        
        Args:
            state: PortfolioState Pydantic model instance
        
        Returns:
            PortfolioSnapshot: The created PortfolioSnapshot record
        """
        return self.log_snapshot(
            total_equity=state.total_equity,
            cash=state.cash
        )
    
    def get_session(self) -> Session:
        """
        Get a database session (for advanced operations).
        
        Returns:
            Session: SQLAlchemy session instance
            
        Note:
            The caller is responsible for closing the session.
            For simple operations, use the log_* methods instead.
        """
        return self.session_factory()

