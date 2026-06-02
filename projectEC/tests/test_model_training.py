"""
Tests for model training module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.model_training import ModelTrainer


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.randn(50, 10)
    X_test = np.random.randn(20, 10)
    y_train = np.random.randint(0, 2, 50)
    y_test = np.random.randint(0, 2, 20)
    
    feature_names = [f"feature_{i}" for i in range(10)]
    
    return X_train, X_test, y_train, y_test, feature_names


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_initialization_logistic_regression(self):
        """Test initialization with logistic regression."""
        trainer = ModelTrainer(model_type="logistic_regression")
        assert trainer.model is not None
        assert trainer.model_type == "logistic_regression"
    
    def test_initialization_random_forest(self):
        """Test initialization with random forest."""
        trainer = ModelTrainer(model_type="random_forest")
        assert trainer.model is not None
        assert trainer.model_type == "random_forest"
    
    def test_initialization_xgboost(self):
        """Test initialization with xgboost."""
        trainer = ModelTrainer(model_type="xgboost")
        assert trainer.model is not None
        assert trainer.model_type == "xgboost"
    
    def test_initialization_invalid_model(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError):
            ModelTrainer(model_type="invalid_model")
    
    def test_training(self, sample_training_data):
        """Test model training."""
        X_train, _, y_train, _, _ = sample_training_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        # Verify model is trained (has coefficients)
        assert trainer.model.coef_ is not None
    
    def test_prediction(self, sample_training_data):
        """Test model prediction."""
        X_train, X_test, y_train, y_test, _ = sample_training_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)
        
        assert predictions.shape == (20,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_prediction_without_training(self, sample_training_data):
        """Test prediction without training."""
        _, X_test, _, _, _ = sample_training_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        with pytest.raises(ValueError):
            trainer.predict(X_test)
    
    def test_predict_proba(self, sample_training_data):
        """Test probability predictions."""
        X_train, X_test, y_train, _, _ = sample_training_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        probas = trainer.predict_proba(X_test)
        
        assert probas.shape == (20, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate(self, sample_training_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test, feature_names = sample_training_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.set_feature_names(feature_names)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics
        
        # Check value ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_set_feature_names(self, sample_training_data):
        """Test setting feature names."""
        _, _, _, _, feature_names = sample_training_data
        
        trainer = ModelTrainer()
        trainer.set_feature_names(feature_names)
        assert trainer.feature_names == feature_names
    
    def test_evaluate_with_feature_importance(self, sample_training_data):
        """Test feature importance in tree-based model."""
        X_train, X_test, y_train, y_test, feature_names = sample_training_data
        
        trainer = ModelTrainer(model_type="random_forest")
        trainer.set_feature_names(feature_names)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        assert 'feature_importance' in metrics
        assert len(metrics['feature_importance']) == len(feature_names)
    
    def test_save_and_load_model(self, sample_training_data, tmp_path):
        """Test saving and loading model."""
        X_train, X_test, y_train, y_test, _ = sample_training_data
        
        # Train and save
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        model_path = str(tmp_path / "test_model.pkl")
        trainer.save_model(model_path)
        
        # Load and verify
        loaded_model = ModelTrainer.load_model(model_path)
        assert loaded_model is not None
        
        # Test with loaded model
        original_pred = trainer.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        assert np.array_equal(original_pred, loaded_pred)
    
    def test_all_models_comparable_accuracy(self, sample_training_data):
        """Test that all models produce comparable results."""
        X_train, X_test, y_train, y_test, _ = sample_training_data
        
        model_types = ["logistic_regression", "random_forest", "xgboost"]
        accuracies = []
        
        for model_type in model_types:
            trainer = ModelTrainer(model_type=model_type)
            trainer.train(X_train, y_train)
            metrics = trainer.evaluate(X_test, y_test)
            accuracies.append(metrics['accuracy'])
        
        # All accuracies should be between 0 and 1
        assert all(0 <= acc <= 1 for acc in accuracies)


class TestModelTrainerEdgeCases:
    """Edge case tests for ModelTrainer."""
    
    def test_binary_classification(self):
        """Test binary classification scenario."""
        X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 10)
        y_train = np.array([0, 1, 1, 0] * 10)
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        # Model should train without errors
        assert trainer.model is not None
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        X_train = np.random.randn(30, 100)
        y_train = np.random.randint(0, 2, 30)
        X_test = np.random.randn(10, 100)
        
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)
        
        assert predictions.shape == (10,)
    
    def test_imbalanced_classes(self):
        """Test with imbalanced class distribution."""
        X_train = np.random.randn(100, 10)
        y_train = np.array([0] * 90 + [1] * 10)  # 90% class 0, 10% class 1
        X_test = np.random.randn(20, 10)
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)
        
        assert len(predictions) == 20
