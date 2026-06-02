"""
Main training script for Heart Disease AI project.
"""

import os
import logging
import joblib
from src.utils import setup_logging, ensure_directory
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
import json


def main():
    """Main training pipeline."""
    
    # Setup logging
    logger = setup_logging("heart_disease")
    logger.info("Starting Heart Disease AI Project")
    
    # Ensure directories exist
    ensure_directory("models")
    ensure_directory("data")
    
    # Load and process data
    logger.info("=" * 50)
    logger.info("DATA PROCESSING PHASE")
    logger.info("=" * 50)
    
    processor = DataProcessor(random_state=42)
    filepath = "data/heart.csv"
    
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        logger.info("Run 'python download_data.py' to download the dataset first.")
        return
    
    X_train, X_test, y_train, y_test, feature_names = processor.process_pipeline(
        filepath=filepath,
        target_column="target",
        test_size=0.2
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Features: {feature_names}")

    preprocessor_path = "models/preprocessor.pkl"
    joblib.dump(processor.transformer, preprocessor_path)
    logger.info(f"Preprocessor pipeline saved to {preprocessor_path}")
    
    # Train models
    logger.info("=" * 50)
    logger.info("MODEL TRAINING PHASE")
    logger.info("=" * 50)
    
    model_types = ["logistic_regression", "random_forest", "xgboost"]
    results = {}
    
    for model_type in model_types:
        logger.info(f"\nTraining {model_type.upper()}...")
        
        trainer = ModelTrainer(model_type=model_type)
        trainer.set_feature_names(feature_names)
        trainer.train(X_train, y_train)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        results[model_type] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
        }
        
        logger.info(f"Results for {model_type}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Save model
        model_path = f"models/{model_type}_model.pkl"
        trainer.save_model(model_path)
    
    # Save results
    results_path = "models/training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    
    # Find best model
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"Best model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
    logger.info("To use the model, run: streamlit run app.py")


if __name__ == "__main__":
    main()
