"""
Unit tests for configuration loader.

Tests valid and invalid YAML configurations to ensure proper validation.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.config.loader import load_config


class TestConfigLoader:
    """Test suite for configuration loader."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        valid_config = {
            "system": {
                "heartbeat_interval": 60,
                "db_url": "sqlite:///test.db"
            },
            "strategies": [
                {
                    "name": "test_strategy",
                    "signal": "momentum",
                    "interval": "1d",
                    "params": {"window": 30, "k": 3}
                }
            ],
            "execution": {
                "provider": "alpaca",
                "paper": True
            },
            "data": {
                "tickers": ["AAPL", "MSFT"],
                "period": "5y",
                "output_path": "data/test.parquet"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            # Verify system config
            assert config.system.heartbeat_interval == 60
            assert config.system.db_url == "sqlite:///test.db"
            
            # Verify strategies
            assert len(config.strategies) == 1
            assert config.strategies[0].name == "test_strategy"
            assert config.strategies[0].signal == "momentum"
            assert config.strategies[0].interval == "1d"
            assert config.strategies[0].params["window"] == 30
            assert config.strategies[0].params["k"] == 3
            
            # Verify execution
            assert config.execution.provider == "alpaca"
            assert config.execution.paper is True
            
            # Verify data
            assert config.data is not None
            assert config.data.tickers == ["AAPL", "MSFT"]
            assert config.data.period == "5y"
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_with_defaults(self):
        """Test loading config with minimal required fields (uses defaults)."""
        minimal_config = {
            "strategies": [
                {
                    "name": "minimal_strategy",
                    "signal": "momentum",
                    "interval": "1d"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            # Should use defaults
            assert config.system.heartbeat_interval == 60  # default
            assert config.execution.provider == "alpaca"  # default
            assert config.execution.paper is True  # default
            assert config.data is None  # optional, not provided
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_yaml_syntax(self):
        """Test loading invalid YAML syntax."""
        invalid_yaml = """
        system:
          heartbeat_interval: 60
          db_url: "sqlite:///test.db"
        strategies:
          - name: "test"
            signal: "momentum"
            interval: "1d"
            params: {window: 30  # Missing closing brace
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises((yaml.YAMLError, ValueError)):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_missing_required_field(self):
        """Test loading config with missing required fields."""
        invalid_config = {
            "system": {
                "heartbeat_interval": 60
                # Missing db_url (has default, so OK)
            },
            "strategies": [
                {
                    "name": "test_strategy",
                    # Missing signal (required)
                    "interval": "1d"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            # Loader wraps ValidationError in ValueError
            with pytest.raises((ValidationError, ValueError)):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_heartbeat_interval(self):
        """Test loading config with invalid heartbeat_interval (<= 0)."""
        invalid_config = {
            "system": {
                "heartbeat_interval": 0,  # Invalid: must be > 0
                "db_url": "sqlite:///test.db"
            },
            "strategies": [
                {
                    "name": "test_strategy",
                    "signal": "momentum",
                    "interval": "1d"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            # Should fail Pydantic validation (Field constraint)
            # Loader wraps ValidationError in ValueError
            with pytest.raises((ValidationError, ValueError), match="heartbeat_interval"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_duplicate_strategy_names(self):
        """Test loading config with duplicate strategy names."""
        invalid_config = {
            "system": {
                "heartbeat_interval": 60,
                "db_url": "sqlite:///test.db"
            },
            "strategies": [
                {
                    "name": "duplicate",
                    "signal": "momentum",
                    "interval": "1d"
                },
                {
                    "name": "duplicate",  # Duplicate name
                    "signal": "momentum",
                    "interval": "1h"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Duplicate strategy names"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_empty_file(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="empty or invalid"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_config_with_multiple_strategies(self):
        """Test loading config with multiple strategies."""
        multi_strategy_config = {
            "system": {
                "heartbeat_interval": 120,
                "db_url": "sqlite:///multi.db"
            },
            "strategies": [
                {
                    "name": "strategy_1",
                    "signal": "momentum",
                    "interval": "1d",
                    "params": {"window": 30}
                },
                {
                    "name": "strategy_2",
                    "signal": "momentum",
                    "interval": "1h",
                    "params": {"window": 7}
                }
            ],
            "execution": {
                "provider": "alpaca",
                "paper": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(multi_strategy_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            assert len(config.strategies) == 2
            assert config.strategies[0].name == "strategy_1"
            assert config.strategies[1].name == "strategy_2"
            assert config.execution.paper is False
            
        finally:
            Path(temp_path).unlink()

