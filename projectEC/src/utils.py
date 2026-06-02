"""
Utility functions for the AI project.
"""

import os
import logging
from typing import Tuple
import numpy as np


def setup_logging(name: str = "heartdisease") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def validate_model_input(features: np.ndarray, n_features: int) -> bool:
    """
    Validate model input features.
    
    Args:
        features: Input features array
        n_features: Expected number of features
        
    Returns:
        True if valid, False otherwise
    """
    if features.shape[-1] != n_features:
        return False
    return True
