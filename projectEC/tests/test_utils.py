"""
Tests for utilities module.
"""

import pytest
import numpy as np
from src.utils import setup_logging, ensure_directory, validate_model_input
import tempfile
import os


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_ensure_directory_creates_new(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test_dir")
            ensure_directory(test_path)
            assert os.path.isdir(test_path)
    
    def test_ensure_directory_existing(self):
        """Test that function works with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise error with existing directory
            ensure_directory(tmpdir)
            assert os.path.isdir(tmpdir)
    
    def test_validate_model_input_valid(self):
        """Test valid model input."""
        features = np.random.randn(10, 5)
        assert validate_model_input(features, 5) is True
    
    def test_validate_model_input_wrong_features(self):
        """Test invalid model input with wrong feature count."""
        features = np.random.randn(10, 5)
        assert validate_model_input(features, 10) is False
    
    def test_validate_model_input_single_sample(self):
        """Test with single sample."""
        features = np.array([[1, 2, 3, 4, 5]])
        assert validate_model_input(features, 5) is True
