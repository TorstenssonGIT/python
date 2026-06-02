"""
Data processing module for the Heart Disease AI project.
Handles data loading, cleaning, and preparation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging


class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataProcessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.data: Optional[pd.DataFrame] = None
        self.numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.categorical_features = [
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
        ]
        self.transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                (
                    'cat',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                    self.categorical_features
                )
            ],
            remainder='drop'
        )
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded dataframe
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not isinstance(filepath, str):
            raise ValueError("Filepath must be a string")
            
        self.data = pd.read_csv(filepath)
        self.logger.info(f"Loaded data with shape: {self.data.shape}")
        return self.data
    
    def explore_data(self) -> dict:
        """
        Perform exploratory data analysis.
        
        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded yet")
        
        stats = {
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict(),
            'describe': self.data.describe().to_dict(),
        }
        
        self.logger.info("Data exploration completed")
        return stats
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean data by handling missing values and normalizing the target.
        
        Returns:
            Cleaned dataframe
        """
        if self.data is None:
            raise ValueError("Data not loaded yet")
        
        # Handle missing values
        initial_rows = len(self.data)
        self.data = self.data.dropna().drop_duplicates().reset_index(drop=True)
        self.logger.info(f"Dropped {initial_rows - len(self.data)} rows with missing values or duplicates")
        
        # Normalize binary target values
        if 'target' in self.data.columns:
            self.data['target'] = self.data['target'].apply(lambda value: 1 if value > 0 else 0)
            self.logger.info(f"Normalized target values to binary classes")
        
        self.logger.info(f"Data shape after cleaning: {self.data.shape}")
        return self.data
    
    def prepare_features_and_target(
        self, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            target_column: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        if self.data is None:
            raise ValueError("Data not loaded yet")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        self.logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.logger.info(
            f"Data split: Train {X_train.shape}, Test {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test
    
    def transform_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Transform features using a preprocessing pipeline.
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed, feature_names)
        """
        X_train_transformed = self.transformer.fit_transform(X_train)
        X_test_transformed = self.transformer.transform(X_test)
        
        encoded_feature_names = self.transformer.named_transformers_['cat'].get_feature_names_out(
            self.categorical_features
        ).tolist()
        feature_names = self.numeric_features + encoded_feature_names
        
        self.logger.info("Features transformed and encoded successfully")
        return X_train_transformed, X_test_transformed, feature_names
    
    def process_pipeline(
        self,
        filepath: str,
        target_column: str,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list]:
        """
        Complete data processing pipeline.
        
        Args:
            filepath: Path to data file
            target_column: Target column name
            test_size: Test set proportion
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed, y_train, y_test, feature_names)
        """
        self.load_data(filepath)
        self.clean_data()
        X, y = self.prepare_features_and_target(target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        X_train_transformed, X_test_transformed, feature_names = self.transform_features(
            X_train,
            X_test
        )
        
        return X_train_transformed, X_test_transformed, y_train, y_test, feature_names
