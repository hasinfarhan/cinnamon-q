"""
Tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.features.build_features import (
    create_numerical_features,
    create_categorical_features,
    select_features
)


class TestFeatureEngineering:
    """Test class for feature engineering functions."""
    
    def test_create_numerical_features(self):
        """Test numerical feature creation."""
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
        
        result = create_numerical_features(test_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)
    
    def test_create_categorical_features(self):
        """Test categorical feature creation."""
        test_data = pd.DataFrame({
            'A': ['x', 'y', 'z', 'x', 'y'],
            'B': [1, 2, 3, 4, 5],
            'C': ['a', 'b', 'c', 'a', 'b']
        })
        
        result = create_categorical_features(test_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)
    
    def test_select_features(self):
        """Test feature selection."""
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = select_features(test_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(test_data.columns)
    
    def test_feature_engineering_pipeline(self):
        """Test the complete feature engineering pipeline."""
        test_data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['a', 'b', 'c', 'a', 'b'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Apply feature engineering steps
        result = create_numerical_features(test_data)
        result = create_categorical_features(result)
        result = select_features(result)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)


if __name__ == "__main__":
    pytest.main([__file__])
