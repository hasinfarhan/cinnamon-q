"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'categorical': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, np.nan, 50, 60, 70, 80, 90, 100],
        'categorical': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_data_large():
    """Create larger sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(10, 2, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path
