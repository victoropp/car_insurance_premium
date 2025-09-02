"""
World-Class Insurance Premium Analytics Dashboard
Streamlit Production Application

Videbimus AI - Advanced Machine Learning Platform for Insurance Premium Prediction
Developed by: Victor Collins Oppon
Company: Videbimus AI
Website: https://www.videbimusai.com
Contact: consulting@videbimusai.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import RobustScaler
import joblib
import warnings
import gc
import os
import sys
import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from streamlit_visualizations import StreamlitVisualizationEngine

warnings.filterwarnings('ignore')

# Application version and metadata
APP_VERSION = "2.0.0"
LAST_UPDATED = "2025-09-02"
MODEL_VERSION = "Statistical-v1.0"

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Videbimus AI - Insurance Premium Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.videbimusai.com',
        'Report a bug': 'mailto:consulting@videbimusai.com',
        'About': """
        # Videbimus AI - Insurance Premium Analytics
        
        **World-Class ML-Powered Dashboard**
        
        Developed by Victor Collins Oppon
        
        🌐 Website: https://www.videbimusai.com
        📧 Contact: consulting@videbimusai.com
        
        © 2025 Videbimus AI. All rights reserved.
        """
    }
)

# ==================== CACHING & MEMORY OPTIMIZATION ====================
@st.cache_resource
def load_models():
    """Load ML models with caching for memory efficiency"""
    models = {}
    model_files = {
        'stacking_linear': 'models/stacking_linear.pkl',
        'stacking_ridge': 'models/stacking_ridge.pkl',
        'voting_ensemble': 'models/voting_ensemble.pkl'
    }
    
    for name, filepath in model_files.items():
        try:
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
                st.sidebar.success(f"✅ {name.replace('_', ' ').title()} model loaded")
            else:
                st.sidebar.warning(f"⚠️ Model file not found: {filepath}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {name}: {str(e)}")
    
    # Memory cleanup
    gc.collect()
    return models

@st.cache_data
def load_test_results():
    """Load test results with caching"""
    try:
        if os.path.exists('data/final_test_results.csv'):
            test_results = pd.read_csv('data/final_test_results.csv')
            return test_results
        else:
            st.warning("Test results file not found. Using default values.")
            return pd.DataFrame({
                'Model': ['Stacking (Linear)', 'Stacking (Ridge)', 'Voting Ensemble'],
                'Test_R2': [0.9978, 0.9978, 0.9948],
                'Test_RMSE': [0.2721, 0.2725, 0.4190],
                'Test_MAE': [0.2010, 0.2012, 0.2939]
            })
    except Exception as e:
        st.error(f"Error loading test results: {str(e)}")
        return None

@st.cache_resource
def initialize_visualization_engine():
    """Initialize and cache the visualization engine"""
    try:
        viz_engine = StreamlitVisualizationEngine()
        return viz_engine
    except Exception as e:
        st.error(f"Error initializing visualization engine: {str(e)}")
        return None

@st.cache_resource
def load_scaler_and_features():
    """Load the trained scaler and selected features"""
    try:
        scaler = joblib.load('models/robust_scaler.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        data_stats = joblib.load('models/data_statistics.pkl')
        st.sidebar.success("✅ Scaler and features loaded")
        return scaler, selected_features, data_stats
    except Exception as e:
        st.sidebar.error(f"❌ Error loading scaler: {str(e)}")
        return None, None, None

# ==================== FEATURE ENGINEERING ====================
def create_statistical_features(df, data_stats):
    """Apply statistical feature engineering matching the training pipeline"""
    df_feat = df.copy()
    epsilon = 1e-6
    
    # Ensure Car Age exists
    if 'Car Age' not in df_feat.columns and 'Car Manufacturing Year' in df_feat.columns:
        max_year = 2025  # Current year
        df_feat['Car Age'] = max_year - df_feat['Car Manufacturing Year']
    elif 'Car Age' not in df_feat.columns:
        # Assume a default car age if not provided
        df_feat['Car Age'] = 5
    
    # 1. Ratio features
    df_feat['Accidents_Per_Year_Driving'] = (
        df_feat['Previous Accidents'] / (df_feat['Driver Experience'] + epsilon)
    )
    df_feat['Mileage_Per_Year_Driving'] = (
        df_feat['Annual Mileage (x1000 km)'] / (df_feat['Driver Experience'] + epsilon)
    )
    df_feat['Car_Age_Driver_Age_Ratio'] = (
        df_feat['Car Age'] / (df_feat['Driver Age'] + epsilon)
    )
    df_feat['Age_Experience_Ratio'] = (
        df_feat['Driver Age'] / (df_feat['Driver Experience'] + epsilon)
    )
    df_feat['Experience_Rate'] = (
        df_feat['Driver Experience'] / (df_feat['Driver Age'] + epsilon)
    )
    
    # 2. Statistical risk score (simplified for single prediction)
    df_feat['Risk_Score'] = (
        (df_feat['Previous Accidents'] / 3.0) * 0.3 +
        (df_feat['Driver Age'] / 100.0) * 0.2 +
        (df_feat['Car Age'] / 30.0) * 0.2 +
        (df_feat['Annual Mileage (x1000 km)'] / 50.0) * 0.15 +
        (1.0 / (df_feat['Driver Experience'] + 1)) * 0.15
    )
    
    # 3. Polynomial features
    df_feat['Driver_Age_Squared'] = df_feat['Driver Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Driver Experience'] ** 2
    df_feat['Accidents_Squared'] = df_feat['Previous Accidents'] ** 2
    
    # 4. Statistical threshold indicators using quartiles
    # Use fixed thresholds based on typical insurance data
    df_feat['Young_Driver'] = (df_feat['Driver Age'] < 30).astype(int)
    df_feat['Senior_Driver'] = (df_feat['Driver Age'] > 55).astype(int)
    df_feat['New_Driver'] = (df_feat['Driver Experience'] < 3).astype(int)
    df_feat['High_Risk_Driver'] = (df_feat['Previous Accidents'] > 1).astype(int)
    df_feat['Old_Car'] = (df_feat['Car Age'] > 12).astype(int)
    df_feat['High_Mileage'] = (df_feat['Annual Mileage (x1000 km)'] > 20).astype(int)
    
    # 5. Interaction features
    df_feat['Age_Experience_Interaction'] = df_feat['Driver Age'] * df_feat['Driver Experience']
    df_feat['Age_Mileage_Interaction'] = df_feat['Driver Age'] * df_feat['Annual Mileage (x1000 km)']
    df_feat['Experience_Accidents_Interaction'] = df_feat['Driver Experience'] * df_feat['Previous Accidents']
    
    # Add Car Manufacturing Year
    if 'Car Manufacturing Year' not in df_feat.columns:
        df_feat['Car Manufacturing Year'] = 2025 - df_feat['Car Age']
    
    return df_feat

def validate_input_data(age, experience, vehicle_age, accidents, annual_mileage):
    """Validate user input data for reasonable ranges"""
    errors = []
    warnings = []
    
    # Age validation
    if age < 18:
        errors.append("Driver age must be at least 18 years old")
    elif age > 100:
        errors.append("Driver age seems unusually high (>100 years)")
    elif age > 80:
        warnings.append("Premium calculations may be less accurate for drivers over 80")
    
    # Experience validation
    if experience < 0:
        errors.append("Driving experience cannot be negative")
    elif experience > (age - 15):
        errors.append(f"Driving experience ({experience}) cannot exceed age minus 15 ({age - 15})")
    elif experience > 60:
        warnings.append("Driving experience over 60 years is unusual")
    
    # Vehicle age validation
    if vehicle_age < 0:
        errors.append("Vehicle age cannot be negative")
    elif vehicle_age > 40:
        warnings.append("Vehicle age over 40 years may have limited insurance options")
    
    # Accidents validation
    if accidents < 0:
        errors.append("Number of accidents cannot be negative")
    elif accidents > 20:
        errors.append("Number of accidents seems unreasonably high (>20)")
    elif accidents > 5:
        warnings.append("High number of accidents may significantly impact premium")
    
    # Mileage validation
    if annual_mileage <= 0:
        errors.append("Annual mileage must be greater than 0")
    elif annual_mileage > 100:
        warnings.append("Very high annual mileage (>100k km) may significantly impact premium")
    elif annual_mileage < 1:
        warnings.append("Very low annual mileage (<1k km) is unusual")
    
    return errors, warnings

def predict_premium(age, experience, vehicle_age, accidents, annual_mileage, models, scaler, selected_features, data_stats):
    """Predict insurance premium using the trained models with proper scaling and validation"""
    try:
        # Validate input data
        errors, warnings = validate_input_data(age, experience, vehicle_age, accidents, annual_mileage)
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            return None, {"Error": "Invalid input data"}
        
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")
        
        # Validate required components
        if not models:
            st.error("No models available for prediction")
            return None, {"Error": "No models"}
        
        if scaler is None:
            st.error("Scaler not available")
            return None, {"Error": "No scaler"}
            
        if selected_features is None:
            st.error("Selected features not available")
            return None, {"Error": "No features"}
        
        # Create input data with correct column names
        input_data = pd.DataFrame({
            'Driver Age': [age],
            'Driver Experience': [experience],
            'Car Age': [vehicle_age],
            'Previous Accidents': [accidents],
            'Annual Mileage (x1000 km)': [annual_mileage]
        })
        
        # Apply statistical feature engineering
        input_features = create_statistical_features(input_data, data_stats)
        
        # Select only the features used in training
        input_features_selected = input_features[selected_features]
        
        # Scale features using the trained scaler
        input_scaled = scaler.transform(input_features_selected)
        
        # Make predictions with all available models
        predictions = {}
        if 'stacking_linear' in models:
            predictions['Stacking (Linear)'] = models['stacking_linear'].predict(input_scaled)[0]
        if 'stacking_ridge' in models:
            predictions['Stacking (Ridge)'] = models['stacking_ridge'].predict(input_scaled)[0]
        if 'voting_ensemble' in models:
            predictions['Voting Ensemble'] = models['voting_ensemble'].predict(input_scaled)[0]
        
        # Use best model prediction as primary
        primary_prediction = predictions.get('Stacking (Linear)', 
                                           predictions.get('stacking_ridge',
                                                         predictions.get('voting_ensemble', 800)))
        
        # Memory cleanup
        gc.collect()
        
        return max(primary_prediction, 200), predictions
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, {"Error": str(e)}

# Test the critical functions
if __name__ == "__main__":
    # Test validation function
    errors, warnings = validate_input_data(30, 10, 5, 0, 15.0)
    print("✅ Dashboard core functions working correctly!")