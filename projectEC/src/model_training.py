"""
Model training module for the Heart Disease AI project.
Handles model creation, training, and evaluation.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
)
import joblib


class ModelTrainer:
    """
    Handles model training, evaluation, and predictions.
    """
    
    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            model_type: Type of model ('logistic_regression', 'random_forest', 'xgboost')
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If model_type is not supported
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
        self._create_model()
    
    def _create_model(self) -> None:
        """Create model based on model_type."""
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear',
                C=1.0,
                class_weight='balanced'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                verbosity=0,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.logger.info(f"Created {self.model_type} model")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred),
        }
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            metrics['feature_importance'] = {
                name: float(imp) for name, imp in zip(self.feature_names, importances)
            }
        
        self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        return joblib.load(filepath)
    
    def set_feature_names(self, feature_names: list) -> None:
        """
        Set feature names for interpretability.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
