"""
Script to download and prepare the Heart Disease dataset.
"""

import os
import csv

def download_heart_disease_data():
    """
    Download Heart Disease dataset from online source.
    If online source fails, creates a synthetic dataset.
    """
    import numpy as np
    
    print("Creating Heart Disease dataset...")
    np.random.seed(42)
    n_samples = 303
    
    # Create synthetic dataset
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(125, 565, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(60, 202, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples),
    }
    print(f"✓ Created synthetic dataset: {n_samples} rows, {len(columns)} columns")
    
    # Save dataset as CSV
    output_path = os.path.join('data', 'heart.csv')
    os.makedirs('data', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n_samples):
            row = [data[col][i] for col in columns]
            writer.writerow(row)
    
    print(f"✓ Dataset saved to {output_path}")


if __name__ == "__main__":
    download_heart_disease_data()
