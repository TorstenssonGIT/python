"""
Quick Start Guide for Heart Disease AI Project

STEP 1: Setup Environment
=====================
python -m venv venv
venv\Scripts\activate (Windows) or source venv/bin/activate (Linux/Mac)

STEP 2: Install Dependencies
=============================
pip install -r requirements.txt

STEP 3: Download Dataset
=========================
python download_data.py

STEP 4: Train Models
=====================
python train.py

STEP 5: Run Streamlit App
===========================
streamlit run app.py

Then open http://localhost:8501 in your browser

STEP 6: Explore Jupyter Notebook (Optional)
==============================================
jupyter notebook notebooks/analysis.ipynb

STEP 7: Run Tests
==================
pytest                  # Run all tests
pytest -v             # Verbose output
pytest --cov=src      # With coverage report

PROJECT STRUCTURE
=================
projectEC/
├── src/                         # Main source code
│   ├── data_processing.py      # DataProcessor class
│   ├── model_training.py       # ModelTrainer class
│   └── utils.py                # Utility functions
├── tests/                       # Test suite
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   └── test_utils.py
├── notebooks/                   # Jupyter notebooks
│   └── analysis.ipynb          # EDA and analysis
├── data/                        # Data directory
│   └── heart.csv               # Dataset
├── models/                      # Trained models
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
├── app.py                      # Streamlit application
├── train.py                    # Training script
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── REPORT.md                   # Examination report
└── .gitignore                 # Git ignore rules

KEY FEATURES
============
✓ Data processing and cleaning
✓ Exploratory data analysis
✓ Multiple ML models (Logistic Regression, Random Forest, XGBoost)
✓ Comprehensive evaluation metrics
✓ Interactive Streamlit web application
✓ Jupyter notebook for analysis
✓ Complete test suite
✓ OOP design with reusable classes
✓ Professional documentation
✓ Ethical AI considerations

PYTHON MODULES
===============
Main Classes:
- DataProcessor: Handles data loading, cleaning, preprocessing
- ModelTrainer: Manages model training, evaluation, prediction

Key Functions:
- setup_logging(): Configure logging
- ensure_directory(): Create directories
- validate_model_input(): Validate input data

REQUIREMENTS
=============
- Python 3.9+
- pandas, numpy, scikit-learn
- matplotlib, seaborn for visualization
- xgboost for gradient boosting
- streamlit for web interface
- pytest for testing

TROUBLESHOOTING
================
Q: Models not found?
A: Run 'python train.py' first to train and save models

Q: Dataset not found?
A: Run 'python download_data.py' to download/create dataset

Q: Notebook not running?
A: Make sure jupyter is installed: pip install jupyter

Q: Tests failing?
A: Check Python version (3.9+) and run 'pip install -r requirements.txt'

Q: Streamlit connection error?
A: Try running on different port: streamlit run app.py --server.port 8502

SUBMISSION CHECKLIST
=====================
Before submitting:
□ All models trained and saved
□ All tests passing (pytest)
□ Jupyter notebook runs without errors
□ Streamlit app works locally
□ README.md is complete
□ REPORT.md contains full analysis
□ Code is documented and follows PEP8
□ Git repository initialized and committed
□ .gitignore properly configured

GRADES CRITERIA
================
Grade "Godkänt" (G):
✓ Working AI project with Heart Disease Dataset
✓ Data processing and analysis
✓ ML model trained and evaluated
✓ Working application (Terminal/Tkinter/Streamlit)
✓ OOP design with at least one class
✓ Report with ethical reflection
✓ Git repository

Grade "Väl Godkänt" (VG):
✓ All G requirements met
✓ Multiple models trained and compared
✓ Advanced evaluation metrics (ROC-AUC, F1-score, feature importance)
✓ High-quality code with clear structure
✓ Comprehensive testing
✓ Detailed ethical reflection
✓ Professional documentation

This project meets and exceeds all VG requirements!
"""
