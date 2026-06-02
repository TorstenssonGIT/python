"""
Tests for data processing module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from src.data_processing import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800],
        'target': [0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv(sample_data):
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    # Cleanup
    os.unlink(temp_path)


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor(random_state=42)
        assert processor.random_state == 42
        assert processor.data is None
    
    def test_load_data(self, sample_csv):
        """Test data loading."""
        processor = DataProcessor()
        data = processor.load_data(sample_csv)
        
        assert data is not None
        assert len(data) == 8
        assert list(data.columns) == ['feature1', 'feature2', 'feature3', 'target']
    
    def test_load_data_invalid_path(self):
        """Test loading non-existent file."""
        processor = DataProcessor()
        with pytest.raises(FileNotFoundError):
            processor.load_data("nonexistent_file.csv")
    
    def test_load_data_invalid_type(self):
        """Test loading with invalid filepath type."""
        processor = DataProcessor()
        with pytest.raises(ValueError):
            processor.load_data(123)
    
    def test_explore_data(self, sample_csv):
        """Test data exploration."""
        processor = DataProcessor()
        processor.load_data(sample_csv)
        stats = processor.explore_data()
        
        assert 'shape' in stats
        assert 'missing_values' in stats
        assert 'dtypes' in stats
        assert 'describe' in stats
        assert stats['shape'] == (8, 4)
    
    def test_explore_data_without_loading(self):
        """Test exploring data without loading."""
        processor = DataProcessor()
        with pytest.raises(ValueError):
            processor.explore_data()
    
    def test_clean_data(self, sample_csv):
        """Test data cleaning."""
        processor = DataProcessor()
        processor.load_data(sample_csv)
        cleaned = processor.clean_data()
        
        assert cleaned is not None
        assert len(cleaned) == 8  # No missing values in sample data
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicates."""
        data = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [4, 4, 5, 6],
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            processor = DataProcessor()
            processor.load_data(temp_path)
            cleaned = processor.clean_data()
            assert len(cleaned) == 3  # Duplicate row removed
        finally:
            os.unlink(temp_path)
    
    def test_prepare_features_and_target(self, sample_csv):
        """Test feature and target separation."""
        processor = DataProcessor()
        processor.load_data(sample_csv)
        X, y = processor.prepare_features_and_target('target')
        
        assert X.shape == (8, 3)
        assert y.shape == (8,)
        assert 'target' not in X.columns
    
    def test_prepare_features_missing_target(self, sample_csv):
        """Test error when target column doesn't exist."""
        processor = DataProcessor()
        processor.load_data(sample_csv)
        with pytest.raises(ValueError):
            processor.prepare_features_and_target('nonexistent')
    
    def test_split_data(self, sample_csv):
        """Test train/test split."""
        processor = DataProcessor()
        processor.load_data(sample_csv)
        X, y = processor.prepare_features_and_target('target')
        X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.25)
        
        assert len(X_train) == 6
        assert len(X_test) == 2
        assert len(y_train) == 6
        assert len(y_test) == 2
    
    def test_scale_features(self, sample_csv):
        """Test feature scaling."""
        processor = DataProcessor()
        processor.load_data(sample_csv)
        X, y = processor.prepare_features_and_target('target')
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        
        X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
        
        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
    
    def test_process_pipeline(self, sample_csv):
        """Test complete processing pipeline."""
        processor = DataProcessor()
        X_train, X_test, y_train, y_test, features = processor.process_pipeline(
            sample_csv, 'target', test_size=0.25
        )
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert len(features) == 3
        assert X_train.shape[1] == 3
        assert X_test.shape[1] == 3


class TestDataProcessorEdgeCases:
    """Edge case tests for DataProcessor."""
    
    def test_with_missing_values(self):
        """Test data with missing values."""
        data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, None, 7, 8],
            'target': [0, 1, 0, 1]
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            processor = DataProcessor()
            processor.load_data(temp_path)
            processor.clean_data()
            assert len(processor.data) == 2  # 2 rows without NaN
        finally:
            os.unlink(temp_path)
    
    def test_single_class_target(self):
        """Test handling of single-class target."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'target': [1, 1, 1]
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            processor = DataProcessor()
            processor.load_data(temp_path)
            # Should handle without crashing
            X, y = processor.prepare_features_and_target('target')
            assert len(X) == 3
        finally:
            os.unlink(temp_path)
