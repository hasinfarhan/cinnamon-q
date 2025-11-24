"""
Configuration Schema - Pydantic Models

This module defines the Pydantic models that validate and structure
the system configuration loaded from YAML files.

Configuration as Code: All hardcoded constants are moved to config.yaml
and validated through these models.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """
    System-level configuration.
    
    Attributes:
        heartbeat_interval: Interval in seconds for the main execution loop
        db_url: Database connection URL (SQLite for MVP)
    """
    heartbeat_interval: int = Field(
        default=60,
        ge=1,
        description="Heartbeat interval in seconds (minimum 1 second)"
    )
    db_url: str = Field(
        default="sqlite:///arynosq.db",
        description="Database connection URL"
    )
    
    model_config = {"frozen": True}


class StrategyConfig(BaseModel):
    """
    Strategy configuration for a single strategy.
    
    Attributes:
        name: Unique name identifier for this strategy
        signal: Signal generator type (e.g., "momentum")
        interval: Execution interval (e.g., "1d" for daily, "1h" for hourly)
        params: Strategy-specific parameters (e.g., window for momentum)
    """
    name: str = Field(description="Unique strategy name identifier")
    signal: str = Field(description="Signal generator type (e.g., 'momentum')")
    interval: str = Field(
        description="Execution interval (e.g., '1d', '1h', '1m')"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters"
    )
    
    model_config = {"frozen": True}


class ExecutionConfig(BaseModel):
    """
    Execution engine configuration.
    
    Attributes:
        provider: Broker provider name (e.g., "alpaca")
        paper: Whether to use paper trading (True) or live trading (False)
    """
    provider: str = Field(
        default="alpaca",
        description="Broker provider name"
    )
    paper: bool = Field(
        default=True,
        description="Use paper trading (True) or live trading (False)"
    )
    
    model_config = {"frozen": True}


class DataConfig(BaseModel):
    """
    Data fetching configuration.
    
    Attributes:
        tickers: List of ticker symbols to fetch
        period: Historical data period (e.g., "5y" for 5 years)
        output_path: Path to save the processed market data
    """
    tickers: List[str] = Field(
        default_factory=list,
        description="List of ticker symbols to fetch"
    )
    period: str = Field(
        default="5y",
        description="Historical data period (e.g., '5y', '1y', '6mo')"
    )
    output_path: str = Field(
        default="data/processed/market_data.parquet",
        description="Path to save the processed market data"
    )
    
    model_config = {"frozen": True}


class OptimizerConfig(BaseModel):
    """
    Portfolio optimizer configuration (Bicameral Brain v2.1 - Sortino Edition).
    
    Attributes:
        lookback_days: Days of history for semicovariance calculation
        benchmark_rate: Benchmark return rate (returns below this are "downside")
        risk_free_rate: Risk-free rate for Sortino ratio calculation
    """
    lookback_days: int = Field(
        default=90,
        ge=7,
        description="Days of history for semicovariance matrix calculation (minimum 7)"
    )
    benchmark_rate: float = Field(
        default=0.0,
        description="Benchmark return rate. Returns below this are considered 'downside' (default: 0.0)"
    )
    risk_free_rate: float = Field(
        default=0.0,
        description="Risk-free rate for Sortino ratio calculation (0.0 for crypto)"
    )
    
    model_config = {"frozen": True}


class AppConfig(BaseModel):
    """
    Root application configuration model.
    
    This is the top-level configuration that contains all subsystem configs.
    It matches the structure of config.yaml.
    
    Attributes:
        system: System-level configuration
        strategies: List of strategy configurations
        execution: Execution engine configuration
        optimizer: Portfolio optimizer configuration (optional)
        data: Data fetching configuration (optional)
    """
    system: SystemConfig = Field(
        default_factory=lambda: SystemConfig(),
        description="System-level configuration"
    )
    strategies: List[StrategyConfig] = Field(
        default_factory=list,
        description="List of strategy configurations"
    )
    execution: ExecutionConfig = Field(
        default_factory=lambda: ExecutionConfig(),
        description="Execution engine configuration"
    )
    optimizer: Optional[OptimizerConfig] = Field(
        default=None,
        description="Portfolio optimizer configuration (optional, for Bicameral Brain)"
    )
    data: Optional[DataConfig] = Field(
        default=None,
        description="Data fetching configuration (optional)"
    )
    
    model_config = {"frozen": True}

