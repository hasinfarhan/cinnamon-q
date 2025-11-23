"""
Unit tests for execution engine domain models.

Tests validate that the Pydantic models correctly enforce system invariants,
specifically the "Halal" constraints (No Leverage, Long Only).
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from execution.models import (
    TargetPortfolio,
    PortfolioState,
    TradeOrder,
    OrderSide,
    OrderType
)


class TestTargetPortfolio:
    """Test suite for TargetPortfolio model validation."""
    
    def test_valid_portfolio(self):
        """Test that a valid portfolio with weights summing to 1.0 is accepted."""
        portfolio = TargetPortfolio(
            weights={"AAPL": 0.5, "GOOGL": 0.3, "MSFT": 0.2}
        )
        assert portfolio.weights["AAPL"] == 0.5
        assert portfolio.weights["GOOGL"] == 0.3
        assert portfolio.weights["MSFT"] == 0.2
        assert sum(portfolio.weights.values()) == 1.0
    
    def test_valid_portfolio_under_allocated(self):
        """Test that a portfolio with weights < 1.0 is accepted (cash reserve)."""
        portfolio = TargetPortfolio(
            weights={"AAPL": 0.4, "GOOGL": 0.3}
        )
        assert sum(portfolio.weights.values()) == 0.7
        assert sum(portfolio.weights.values()) < 1.0
    
    def test_valid_portfolio_with_epsilon(self):
        """Test that floating point precision within epsilon is accepted."""
        # Sum is 1.0 + 0.5e-6 (within tolerance)
        portfolio = TargetPortfolio(
            weights={"AAPL": 0.5, "GOOGL": 0.3, "MSFT": 0.2 + 0.5e-6}
        )
        total = sum(portfolio.weights.values())
        assert total > 1.0
        assert total <= 1.0 + 1e-6  # Should pass
    
    def test_rejects_leverage_exceeds_one(self):
        """Test that portfolio with sum > 1.0 + epsilon is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TargetPortfolio(
                weights={"AAPL": 0.6, "GOOGL": 0.5}  # Sum = 1.1
            )
        
        error_message = str(exc_info.value)
        assert "No Leverage constraint violated" in error_message
        assert "exceeds 1.0" in error_message
    
    def test_rejects_leverage_with_epsilon(self):
        """Test that portfolio exceeding 1.0 + epsilon is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TargetPortfolio(
                weights={"AAPL": 0.5, "GOOGL": 0.3, "MSFT": 0.2 + 2e-6}  # Sum > 1.0 + 1e-6
            )
        
        error_message = str(exc_info.value)
        assert "No Leverage constraint violated" in error_message
    
    def test_rejects_negative_weights(self):
        """Test that negative weights are rejected (Long Only constraint)."""
        with pytest.raises(ValidationError) as exc_info:
            TargetPortfolio(
                weights={"AAPL": 0.5, "GOOGL": -0.1}  # Negative weight
            )
        
        error_message = str(exc_info.value)
        assert "Negative weight not allowed" in error_message
        assert "Long Only constraint" in error_message
        assert "GOOGL" in error_message
    
    def test_rejects_multiple_negative_weights(self):
        """Test that multiple negative weights are caught."""
        with pytest.raises(ValidationError) as exc_info:
            TargetPortfolio(
                weights={"AAPL": -0.1, "GOOGL": -0.2}
            )
        
        error_message = str(exc_info.value)
        assert "Negative weight not allowed" in error_message
    
    def test_empty_portfolio(self):
        """Test that an empty portfolio (all cash) is valid."""
        portfolio = TargetPortfolio(weights={})
        assert len(portfolio.weights) == 0
        assert sum(portfolio.weights.values()) == 0.0
    
    def test_portfolio_immutability(self):
        """Test that portfolio model fields are immutable (frozen=True)."""
        portfolio = TargetPortfolio(weights={"AAPL": 0.5})
        
        # In Pydantic v2, frozen models raise ValidationError when trying to modify fields
        with pytest.raises(ValidationError) as exc_info:
            # Try to modify a top-level field (should fail due to frozen=True)
            portfolio.timestamp = datetime(2025, 1, 1)
        
        error_message = str(exc_info.value)
        assert "frozen" in error_message.lower() or "Instance is frozen" in error_message
    
    def test_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated if not provided."""
        portfolio = TargetPortfolio(weights={"AAPL": 0.5})
        assert isinstance(portfolio.timestamp, datetime)
    
    def test_timestamp_can_be_set(self):
        """Test that timestamp can be explicitly set."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        portfolio = TargetPortfolio(
            timestamp=custom_time,
            weights={"AAPL": 0.5}
        )
        assert portfolio.timestamp == custom_time


class TestPortfolioState:
    """Test suite for PortfolioState model validation."""
    
    def test_valid_state(self):
        """Test that a valid portfolio state is accepted."""
        state = PortfolioState(
            cash=10000.0,
            total_equity=50000.0,
            positions={"AAPL": 10.0, "GOOGL": 5.0},
            valuations={"AAPL": 1500.0, "GOOGL": 500.0}
        )
        assert state.cash == 10000.0
        assert state.total_equity == 50000.0
        assert state.positions["AAPL"] == 10.0
        assert state.positions["GOOGL"] == 5.0
        assert state.valuations["AAPL"] == 1500.0
        assert state.valuations["GOOGL"] == 500.0
    
    def test_rejects_negative_cash(self):
        """Test that negative cash is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioState(cash=-100.0, total_equity=1000.0, valuations={})
        
        error_message = str(exc_info.value)
        assert "greater than or equal to 0" in error_message.lower()
    
    def test_rejects_negative_equity(self):
        """Test that negative total equity is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioState(cash=1000.0, total_equity=-100.0, valuations={})
        
        error_message = str(exc_info.value)
        assert "greater than or equal to 0" in error_message.lower()
    
    def test_allows_zero_cash(self):
        """Test that zero cash is allowed."""
        state = PortfolioState(cash=0.0, total_equity=1000.0, valuations={})
        assert state.cash == 0.0
    
    def test_allows_zero_equity(self):
        """Test that zero equity is allowed."""
        state = PortfolioState(cash=0.0, total_equity=0.0, valuations={})
        assert state.total_equity == 0.0
    
    def test_empty_positions(self):
        """Test that empty positions dictionary is valid."""
        state = PortfolioState(
            cash=10000.0,
            total_equity=10000.0,
            positions={},
            valuations={}
        )
        assert len(state.positions) == 0
        assert len(state.valuations) == 0
    
    def test_state_immutability(self):
        """Test that portfolio state is immutable (frozen=True)."""
        state = PortfolioState(cash=1000.0, total_equity=2000.0, valuations={})
        
        with pytest.raises(ValidationError):
            # Try to modify cash (should fail due to frozen=True)
            state.cash = 2000.0
    
    def test_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated if not provided."""
        state = PortfolioState(cash=1000.0, total_equity=2000.0, valuations={})
        assert isinstance(state.timestamp, datetime)


class TestTradeOrder:
    """Test suite for TradeOrder model validation."""
    
    def test_valid_market_order_with_qty(self):
        """Test that a valid market order with quantity is accepted."""
        order = TradeOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=10.0
        )
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.qty == 10.0
        assert order.notional is None
    
    def test_valid_market_order_with_notional(self):
        """Test that a valid market order with notional is accepted."""
        order = TradeOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional=1000.0
        )
        assert order.notional == 1000.0
        assert order.qty is None
    
    def test_valid_limit_order(self):
        """Test that a valid limit order is accepted."""
        order = TradeOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=10.0,
            limit_price=150.0
        )
        assert order.limit_price == 150.0
    
    def test_rejects_no_qty_or_notional(self):
        """Test that order without qty or notional is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )
        
        error_message = str(exc_info.value)
        assert "Either 'qty' or 'notional' must be set" in error_message
    
    def test_rejects_both_qty_and_notional(self):
        """Test that order with both qty and notional is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=10.0,
                notional=1000.0
            )
        
        error_message = str(exc_info.value)
        assert "Cannot set both 'qty' and 'notional'" in error_message
    
    def test_rejects_limit_order_without_price(self):
        """Test that limit order without limit_price is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                qty=10.0
            )
        
        error_message = str(exc_info.value)
        assert "'limit_price' is required when 'order_type' is LIMIT" in error_message
    
    def test_client_order_id_preserved(self):
        """Test that client_order_id is preserved for idempotency."""
        order = TradeOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=10.0,
            client_order_id="test-id-123"
        )
        assert order.client_order_id == "test-id-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

