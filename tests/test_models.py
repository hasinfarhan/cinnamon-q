"""
Tests for model training functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.train_model import (
    split_data,
    train_model,
    evaluate_model,
    cross_validate_model
)


class TestModelTraining:
    """Test class for model training functions."""
    
    def test_split_data(self):
        """Test data splitting function."""
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        X_train, X_test, y_train, y_test = split_data(test_data, 'target', test_size=0.2)
        
        # Check that splits are correct size
        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2
        
        # Check that target column is not in features
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
    
    def test_train_model(self):
        """Test model training function."""
        # Create test data
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        y_train = pd.Series([0, 1, 0, 1, 0])
        
        model_params = {
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = train_model(X_train, y_train, model_params)
        
        # Check that model is trained
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        y_test = pd.Series([0, 1, 0, 1])
        
        # Create a simple trained model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)
        
        metrics = evaluate_model(model, X_test, y_test)
        
        # Check that metrics are returned
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        assert isinstance(metrics['accuracy'], float)
    
    def test_cross_validate_model(self):
        """Test cross-validation function."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create test data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        cv_results = cross_validate_model(model, X, y, cv=3)
        
        # Check that CV results are returned
        assert 'cv_scores' in cv_results
        assert 'cv_mean' in cv_results
        assert 'cv_std' in cv_results
        assert len(cv_results['cv_scores']) == 3
        assert isinstance(cv_results['cv_mean'], float)


if __name__ == "__main__":
    pytest.main([__file__])
