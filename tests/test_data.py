"""
Tests for data processing functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.make_dataset import clean_data, preprocess_data


class TestDataProcessing:
    """Test class for data processing functions."""
    
    def test_clean_data(self):
        """Test data cleaning function."""
        # Create test data with missing values and duplicates
        test_data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 1],
            'B': ['x', 'y', 'z', 'x', 'x'],
            'C': [1.1, 2.2, 3.3, 4.4, 1.1]
        })
        
        # Test that function runs without error
        result = clean_data(test_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_preprocess_data(self):
        """Test data preprocessing function."""
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['x', 'y', 'z', 'x', 'y'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        # Test that function runs without error
        result = preprocess_data(test_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_data_types(self):
        """Test that data processing maintains appropriate data types."""
        test_data = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['a', 'b', 'c'],
            'float': [1.1, 2.2, 3.3]
        })
        
        result = preprocess_data(test_data)
        assert result.dtypes['numeric'] in [np.int64, np.float64]
        assert result.dtypes['float'] == np.float64


if __name__ == "__main__":
    pytest.main([__file__])
