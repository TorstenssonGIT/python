import os
import numpy as np
from src.model_training import ModelTrainer

model_names = ['logistic_regression', 'random_forest', 'xgboost']
for model_name in model_names:
    path = os.path.join('models', f'{model_name}_model.pkl')
    model = ModelTrainer.load_model(path)
    sample = np.array([[50, 1, 1, 130, 240, 0, 1, 150, 0, 1.0, 1, 0, 2]])
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    print(f'{model_name}: pred={pred}, proba=[{proba[0]:.3f},{proba[1]:.3f}]')
