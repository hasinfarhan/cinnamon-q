"""
Database Models - ORM Definitions

This module defines the SQLAlchemy ORM models for Trade and PortfolioSnapshot.
These models represent the persistent state of the trading bot.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Trade(Base):
    """
    Trade Model - Records every executed trade.
    
    This table serves as the "Ledger of Truth" for all trading activity.
    It enables audit trails, performance tracking, and idempotency checks.
    
    Attributes:
        id: Primary key (auto-increment)
        timestamp: UTC timestamp when trade was executed
        strategy_name: Name of the strategy that generated this trade
        symbol: Trading symbol (e.g., "AAPL", "BTC/USD")
        side: Order side ("buy" or "sell")
        qty: Quantity of shares/coins traded
        price: Average fill price per unit
        notional: Total USD value of the trade (qty * price)
        client_order_id: Unique identifier for idempotency (unique constraint)
    """
    
    __tablename__ = "trades"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # "buy" or "sell"
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    notional: Mapped[float] = mapped_column(Float, nullable=False)
    client_order_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )
    
    __table_args__ = (
        UniqueConstraint('client_order_id', name='uq_trade_client_order_id'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side}, "
            f"qty={self.qty}, price={self.price}, notional={self.notional}, "
            f"timestamp={self.timestamp})>"
        )


class PortfolioSnapshot(Base):
    """
    Portfolio Snapshot Model - Periodic records of portfolio state.
    
    This table tracks portfolio value over time, enabling performance
    calculation (Sortino Ratio, Sharpe Ratio, etc.) independent of the broker.
    
    Attributes:
        id: Primary key (auto-increment)
        timestamp: UTC timestamp when snapshot was taken
        total_equity: Total portfolio value in USD
        cash: Available cash balance in USD
    """
    
    __tablename__ = "portfolio_snapshots"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    total_equity: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    
    def __repr__(self) -> str:
        return (
            f"<PortfolioSnapshot(id={self.id}, timestamp={self.timestamp}, "
            f"total_equity={self.total_equity}, cash={self.cash})>"
        )

