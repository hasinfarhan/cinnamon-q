"""
Configuration Loader

This module provides functionality to load and validate configuration from YAML files.
Implements Configuration as Code for the entire system.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .schema import AppConfig


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load and validate configuration from a YAML file.
    
    This function:
    1. Loads the YAML file from the specified path (or default)
    2. Parses it into a dictionary
    3. Validates it against the AppConfig Pydantic model
    4. Returns a validated, immutable AppConfig instance
    
    Args:
        config_path: Path to the config.yaml file.
            If None, searches for 'config.yaml' in:
            - Current directory
            - Project root (parent of src/)
            - Scripts directory (for run_bot.py)
    
    Returns:
        AppConfig: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        ValidationError: If configuration doesn't match schema
        ValueError: If validation constraints are violated (e.g., heartbeat <= 0)
    """
    # Find config file
    if config_path is None:
        # Try multiple locations
        possible_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
            Path(__file__).parent.parent.parent / "scripts" / "config.yaml",
        ]
        
        config_file = None
        for path in possible_paths:
            if path.exists():
                config_file = path
                break
        
        if config_file is None:
            raise FileNotFoundError(
                f"Config file not found. Searched in:\n"
                + "\n".join(f"  - {p}" for p in possible_paths)
                + "\n\nPlease create config.yaml or specify a path."
            )
    else:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Load YAML
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_file}: {e}")
    except Exception as e:
        raise IOError(f"Error reading config file {config_file}: {e}")
    
    if config_dict is None:
        raise ValueError(f"Config file {config_file} is empty or invalid")
    
    # Validate and parse into Pydantic model
    try:
        config = AppConfig(**config_dict)
    except ValidationError as e:
        # Format validation errors nicely
        error_messages = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_messages.append(f"  {field}: {msg}")
        
        # Re-raise with formatted message (Pydantic v2 compatible)
        raise ValueError(
            f"Configuration validation failed:\n" + "\n".join(error_messages)
        ) from e
    
    # Additional custom validation
    _validate_config(config)
    
    return config


def _validate_config(config: AppConfig) -> None:
    """
    Perform additional custom validation beyond Pydantic schema.
    
    This function enforces business logic constraints that may not be
    expressible in the Pydantic schema alone.
    
    Args:
        config: The AppConfig instance to validate
        
    Raises:
        ValueError: If validation constraints are violated
    """
    # Validate heartbeat interval
    if config.system.heartbeat_interval <= 0:
        raise ValueError(
            f"heartbeat_interval must be > 0, got {config.system.heartbeat_interval}"
        )
    
    # Validate strategy names are unique
    strategy_names = [s.name for s in config.strategies]
    if len(strategy_names) != len(set(strategy_names)):
        duplicates = [name for name in strategy_names if strategy_names.count(name) > 1]
        raise ValueError(
            f"Duplicate strategy names found: {set(duplicates)}. "
            "Each strategy must have a unique name."
        )
    
    # Validate signal types (basic check)
    valid_signals = ['momentum']  # Can be extended
    for strategy in config.strategies:
        if strategy.signal not in valid_signals:
            # Warning, not error - allow extensibility
            print(
                f"⚠ Warning: Unknown signal type '{strategy.signal}' "
                f"for strategy '{strategy.name}'. Valid types: {valid_signals}"
            )
    
    # Validate intervals (basic format check)
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    for strategy in config.strategies:
        if strategy.interval not in valid_intervals:
            print(
                f"⚠ Warning: Unusual interval '{strategy.interval}' "
                f"for strategy '{strategy.name}'. Common intervals: {valid_intervals}"
            )

