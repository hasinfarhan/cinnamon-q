"""
Unit tests for database layer.

Tests database models, service, and persistence operations.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.database.base import Base, get_engine, get_session_factory
from src.database.models import PortfolioSnapshot, Trade
from src.database.service import DatabaseService


class TestDatabaseModels:
    """Test database model definitions."""
    
    def test_trade_model_structure(self):
        """Test that Trade model has all required fields."""
        # Check that Trade has all expected columns
        assert hasattr(Trade, 'id')
        assert hasattr(Trade, 'timestamp')
        assert hasattr(Trade, 'strategy_name')
        assert hasattr(Trade, 'symbol')
        assert hasattr(Trade, 'side')
        assert hasattr(Trade, 'qty')
        assert hasattr(Trade, 'price')
        assert hasattr(Trade, 'notional')
        assert hasattr(Trade, 'client_order_id')
    
    def test_portfolio_snapshot_model_structure(self):
        """Test that PortfolioSnapshot model has all required fields."""
        # Check that PortfolioSnapshot has all expected columns
        assert hasattr(PortfolioSnapshot, 'id')
        assert hasattr(PortfolioSnapshot, 'timestamp')
        assert hasattr(PortfolioSnapshot, 'total_equity')
        assert hasattr(PortfolioSnapshot, 'cash')
    
    def test_trade_table_name(self):
        """Test that Trade table has correct name."""
        assert Trade.__tablename__ == "trades"
    
    def test_portfolio_snapshot_table_name(self):
        """Test that PortfolioSnapshot table has correct name."""
        assert PortfolioSnapshot.__tablename__ == "portfolio_snapshots"


class TestDatabaseService:
    """Test DatabaseService operations."""
    
    def test_service_initialization(self):
        """Test that DatabaseService initializes correctly."""
        # Use in-memory database for testing
        db = DatabaseService("sqlite:///:memory:")
        
        assert db.engine is not None
        assert db.session_factory is not None
    
    def test_service_creates_tables(self):
        """Test that DatabaseService creates tables on initialization."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Check that tables exist
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        assert "trades" in tables
        assert "portfolio_snapshots" in tables
    
    def test_log_trade(self):
        """Test logging a trade to the database."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Create a test trade
        trade = db.log_trade(
            strategy_name="test_strategy",
            symbol="AAPL",
            side="buy",
            qty=10.0,
            price=150.0,
            notional=1500.0,
            client_order_id="test_order_123"
        )
        
        # Verify trade was created
        assert trade.id is not None
        assert trade.strategy_name == "test_strategy"
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.qty == 10.0
        assert trade.price == 150.0
        assert trade.notional == 1500.0
        assert trade.client_order_id == "test_order_123"
        assert trade.timestamp is not None
        assert trade.timestamp.tzinfo == timezone.utc  # UTC timezone
    
    def test_log_trade_retrieval(self):
        """Test retrieving a logged trade from the database."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Log a trade
        trade = db.log_trade(
            strategy_name="test_strategy",
            symbol="BTC/USD",
            side="sell",
            qty=0.5,
            price=50000.0,
            notional=25000.0,
            client_order_id="test_order_456"
        )
        
        # Retrieve it using a new session
        with db.get_session() as session:
            retrieved = session.query(Trade).filter_by(
                client_order_id="test_order_456"
            ).first()
            
            assert retrieved is not None
            assert retrieved.id == trade.id
            assert retrieved.symbol == "BTC/USD"
            assert retrieved.side == "sell"
            assert retrieved.qty == 0.5
            assert retrieved.price == 50000.0
            assert retrieved.notional == 25000.0
    
    def test_log_trade_from_dict(self):
        """Test logging a trade from a dictionary."""
        db = DatabaseService("sqlite:///:memory:")
        
        trade_data = {
            "strategy_name": "momentum_bluechip",
            "symbol": "MSFT",
            "side": "buy",
            "qty": 5.0,
            "price": 300.0,
            "notional": 1500.0,
            "client_order_id": "dict_order_789"
        }
        
        trade = db.log_trade_from_dict(trade_data)
        
        assert trade.symbol == "MSFT"
        assert trade.side == "buy"
        assert trade.qty == 5.0
        assert trade.client_order_id == "dict_order_789"
    
    def test_log_trade_duplicate_client_order_id(self):
        """Test that duplicate client_order_id raises IntegrityError."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Log first trade
        db.log_trade(
            strategy_name="test",
            symbol="AAPL",
            side="buy",
            qty=1.0,
            price=100.0,
            notional=100.0,
            client_order_id="duplicate_id"
        )
        
        # Try to log duplicate
        with pytest.raises(IntegrityError):
            db.log_trade(
                strategy_name="test",
                symbol="AAPL",
                side="buy",
                qty=1.0,
                price=100.0,
                notional=100.0,
                client_order_id="duplicate_id"  # Same ID
            )
    
    def test_log_snapshot(self):
        """Test logging a portfolio snapshot."""
        db = DatabaseService("sqlite:///:memory:")
        
        snapshot = db.log_snapshot(
            total_equity=100000.0,
            cash=50000.0
        )
        
        # Verify snapshot was created
        assert snapshot.id is not None
        assert snapshot.total_equity == 100000.0
        assert snapshot.cash == 50000.0
        assert snapshot.timestamp is not None
        assert snapshot.timestamp.tzinfo == timezone.utc  # UTC timezone
    
    def test_log_snapshot_retrieval(self):
        """Test retrieving a logged snapshot from the database."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Log a snapshot
        snapshot = db.log_snapshot(
            total_equity=150000.0,
            cash=75000.0
        )
        
        # Retrieve it using a new session
        with db.get_session() as session:
            retrieved = session.query(PortfolioSnapshot).filter_by(
                id=snapshot.id
            ).first()
            
            assert retrieved is not None
            assert retrieved.total_equity == 150000.0
            assert retrieved.cash == 75000.0
            # SQLite returns naive datetime, but time components should match
            # (timezone is normalized by service when returned from log_snapshot)
            assert retrieved.timestamp.replace(tzinfo=None) == snapshot.timestamp.replace(tzinfo=None)
    
    def test_log_snapshot_from_state(self):
        """Test logging a snapshot from PortfolioState Pydantic model."""
        # Import PortfolioState
        from execution.models import PortfolioState
        
        db = DatabaseService("sqlite:///:memory:")
        
        # Create a PortfolioState
        state = PortfolioState(
            cash=25000.0,
            total_equity=125000.0,
            positions={"AAPL": 10.0},
            valuations={"AAPL": 100000.0}
        )
        
        # Log snapshot from state
        snapshot = db.log_snapshot_from_state(state)
        
        assert snapshot.total_equity == 125000.0
        assert snapshot.cash == 25000.0
    
    def test_multiple_trades(self):
        """Test logging multiple trades."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Log multiple trades
        trade1 = db.log_trade(
            strategy_name="strategy1",
            symbol="AAPL",
            side="buy",
            qty=10.0,
            price=150.0,
            notional=1500.0,
            client_order_id="order1"
        )
        
        trade2 = db.log_trade(
            strategy_name="strategy1",
            symbol="MSFT",
            side="buy",
            qty=5.0,
            price=300.0,
            notional=1500.0,
            client_order_id="order2"
        )
        
        # Retrieve all trades
        with db.get_session() as session:
            all_trades = session.query(Trade).all()
            
            assert len(all_trades) == 2
            assert trade1.id != trade2.id
    
    def test_timestamp_timezone(self):
        """Test that timestamps are stored in UTC."""
        db = DatabaseService("sqlite:///:memory:")
        
        # Log trade with explicit UTC timestamp
        explicit_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        trade = db.log_trade(
            strategy_name="test",
            symbol="AAPL",
            side="buy",
            qty=1.0,
            price=100.0,
            notional=100.0,
            client_order_id="timezone_test",
            timestamp=explicit_time
        )
        
        # SQLite strips timezone on storage, but service restores it
        # So we check that the timezone is UTC and the time components match
        assert trade.timestamp.tzinfo == timezone.utc
        assert trade.timestamp.replace(tzinfo=None) == explicit_time.replace(tzinfo=None)
        
        # Log snapshot with default timestamp (should be UTC)
        snapshot = db.log_snapshot(
            total_equity=100000.0,
            cash=50000.0
        )
        
        assert snapshot.timestamp.tzinfo == timezone.utc

