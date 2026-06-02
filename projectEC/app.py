"""
Streamlit application for Heart Disease Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.model_training import ModelTrainer
from src.utils import setup_logging
import json
import os

INPUT_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


# Setup logging
logger = setup_logging("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .css-18e3th9 { padding-top: 1rem; }
    .main { padding: 1.5rem; }
    .block-container { padding: 1.5rem 2rem 2rem 2rem; }
    .stApp { background-color: #e7f6e7 !important; }
    .stApp, .stApp * { color: #0f172a !important; }
    .stApp input, .stApp select, .stApp textarea, .stApp button { color: #0f172a !important; }
    .stButton>button { background-color: #f8d7da !important; color: #0f172a !important; border: 1px solid #dc3545 !important; }
    .stButton>button:hover { background-color: #f1c0c4 !important; }
    .stMetric { background-color: #d4edda !important; padding: 1rem !important; border-radius: 0.65rem !important; border: 1px solid #c3e6cb !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #102a43 !important; }
    .stSidebar .css-1d391kg { background-color: #dceefe !important; border-right: 1px solid #c7d8ed !important; }
    .css-1dp5vir, .css-ffhzg2 { color: #0f172a !important; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSlider>div>div>input, .stSelectbox>div>div>div>div { color: #0f172a !important; background-color: #e5f2ff !important; }
    .css-ffhzg2 { background-color: #dceefe !important; }
    .element-container { border-radius: 0.75rem !important; }
    .stAlert, .stWarning, .stError, .stSuccess { background-color: #f8d7da !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load trained model."""
    try:
        return ModelTrainer.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_preprocessor():
    """Load preprocessing pipeline for prediction."""
    try:
        return joblib.load("models/preprocessor.pkl")
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None


@st.cache_data
def load_results():
    """Load training results."""
    try:
        with open("models/training_results.json", 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    """Main Streamlit application."""
    
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown("**AI-Powered Prediction System Using Machine Learning**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "🔮 Prediction":
        show_prediction()
    elif page == "📊 Model Performance":
        show_performance()
    elif page == "ℹ️ About":
        show_about()


def show_home():
    """Home page."""
    st.markdown("""
    ## Welcome to the Heart Disease Risk Predictor
    
    This application uses **machine learning models** to predict the risk of heart disease
    based on medical indicators.
    
    ### Key Features
    - 🤖 Multiple ML models (Logistic Regression, Random Forest, XGBoost)
    - 📊 Detailed patient input interface
    - 📈 Real-time predictions with confidence scores
    - 📉 Model performance analytics
    
    ### How to Use
    1. Navigate to **Prediction** in the sidebar
    2. Enter your medical parameters
    3. Click "Make Prediction"
    4. View your risk assessment
    
    ### About the Dataset
    - **Source**: UCI Heart Disease Dataset
    - **Samples**: 303 patients
    - **Features**: 13 medical indicators
    - **Target**: Heart disease presence (0 = No, 1 = Yes)
    """)


def show_prediction():
    """Prediction page."""
    st.header("🔮 Make a Prediction")
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["random_forest", "logistic_regression", "xgboost"],
            help="Choose which ML model to use for prediction"
        )
    
    # Load model
    model_path = f"models/{model_choice}_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model not found. Please train models first by running `python train.py`")
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    st.markdown("---")
    st.markdown("### Patient Medical Parameters")
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (years)", min_value=29, max_value=77, value=50)
        sex = st.selectbox("Sex", ["Female (0)", "Male (1)"], help="0 = Female, 1 = Male")
        cp = st.slider("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
        trestbps = st.slider("Resting Blood Pressure (mmHg)", min_value=90, max_value=200, value=130)
    
    with col2:
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=125, max_value=565, value=240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
        restecg = st.slider("Resting ECG (0-2)", min_value=0, max_value=2, value=1)
        thalach = st.slider("Max Heart Rate Achieved", min_value=60, max_value=202, value=150)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        oldpeak = st.slider("ST Depression (0-6.2)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope = st.slider("ST Slope (0-2)", min_value=0, max_value=2, value=1)
        ca = st.slider("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.slider("Thalassemia (0-3)", min_value=0, max_value=3, value=2)
    
    # Prepare input
    sex_val = int(sex.split("(")[1].strip(")"))
    fbs_val = int(fbs.split("(")[1].strip(")"))
    exang_val = int(exang.split("(")[1].strip(")"))
    
    input_data = np.array([[
        age, sex_val, cp, trestbps, chol, fbs_val, restecg,
        thalach, exang_val, oldpeak, slope, ca, thal
    ]])
    
    # Make prediction
    if st.button("🎯 Make Prediction", use_container_width=True):
        try:
            preprocessor = load_preprocessor()
            if preprocessor is None:
                return

            input_df = pd.DataFrame(input_data, columns=INPUT_COLUMNS)
            input_transformed = preprocessor.transform(input_df)
            
            prediction = model.predict(input_transformed)[0]
            probability = model.predict_proba(input_transformed)[0]
            
            st.markdown("---")
            st.markdown("### Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
                st.metric("Risk Assessment", risk_level)
            
            with col2:
                st.metric("Disease Probability", f"{probability[1]*100:.1f}%")
            
            with col3:
                st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
            
            # Detailed explanation
            st.markdown("#### Interpretation")
            if prediction == 1:
                st.warning(
                    f"⚠️ The model predicts a **{probability[1]*100:.1f}% probability** of heart disease. "
                    f"Please consult with a healthcare professional for proper diagnosis."
                )
            else:
                st.success(
                    f"✅ The model predicts a **{probability[0]*100:.1f}% probability** of no heart disease. "
                    f"Continue maintaining healthy habits!"
                )
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def show_performance():
    """Model performance page."""
    st.header("📊 Model Performance")
    
    results = load_results()
    if results is None:
        st.error("No training results found. Please run `python train.py` first.")
        return
    
    st.markdown("### Training Results Comparison")
    
    # Convert results to dataframe
    df = pd.DataFrame(results).T
    df = df.round(4)
    
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accuracy Comparison")
        st.bar_chart(df['accuracy'])
    
    with col2:
        st.markdown("#### ROC-AUC Comparison")
        st.bar_chart(df['roc_auc'])
    
    st.markdown("#### Detailed Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Precision")
        st.bar_chart(df['precision'])
    with col2:
        st.markdown("##### Recall")
        st.bar_chart(df['recall'])
    with col3:
        st.markdown("##### F1 Score")
        st.bar_chart(df['f1'])


def show_about():
    """About page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Project Overview
    This Heart Disease Risk Predictor is an AI/ML project developed as part of
    a Python Programming with AI course examination.
    
    ### Technologies Used
    - **Python 3.9+**
    - **Streamlit**: Web application framework
    - **Scikit-learn**: Machine learning models
    - **Pandas/NumPy**: Data processing
    - **XGBoost**: Gradient boosting model
    - **Matplotlib/Seaborn**: Data visualization
    
    ### Machine Learning Models
    1. **Logistic Regression**: Baseline classification model
    2. **Random Forest**: Ensemble method with feature importance
    3. **XGBoost**: Gradient boosting for improved performance
    
    ### Dataset
    - **Heart Disease Dataset** from UCI Machine Learning Repository
    - **Pattern**: Binary classification (presence/absence of disease)
    - **Features**: 13 medical indicators including:
        - Demographic: Age, Sex
        - Cardiovascular: Blood Pressure, Cholesterol, Heart Rate
        - Clinical: Chest pain type, ECG results, exercise-induced angina
    
    ### Key Features
    ✅ Multiple ML models for comparison
    ✅ Interactive prediction interface
    ✅ Model performance analytics
    ✅ Production-ready code structure
    ✅ Comprehensive documentation
    
    ### Ethical Considerations
    ⚠️ **Important**: This tool is for educational purposes and demonstration only.
    It should **NOT** be used for actual medical diagnosis without professional validation.
    
    - Predictions are based on statistical patterns in the dataset
    - Real medical decisions require professional medical expertise
    - Dataset limitations and potential biases should be considered
    - Privacy and data security are paramount when handling health information
    
    ### Author
    Student Project - Python Programming with AI Course
    
    ### License
    Educational Use Only
    """)


if __name__ == "__main__":
    main()
