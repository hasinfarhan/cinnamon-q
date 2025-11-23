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


class AppConfig(BaseModel):
    """
    Root application configuration model.
    
    This is the top-level configuration that contains all subsystem configs.
    It matches the structure of config.yaml.
    
    Attributes:
        system: System-level configuration
        strategies: List of strategy configurations
        execution: Execution engine configuration
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
    
    model_config = {"frozen": True}

