import sys
import os
sys.path.append('streamlit_dashboard')

import joblib
import pandas as pd
import numpy as np

# Load models and scaler
model = joblib.load('streamlit_dashboard/models/stacking_linear.pkl')
scaler = joblib.load('streamlit_dashboard/models/robust_scaler.pkl')
selected_features = joblib.load('streamlit_dashboard/models/selected_features.pkl')
data_stats = joblib.load('streamlit_dashboard/models/data_statistics.pkl')

print(f"Model expects: {model.n_features_in_} features")
print(f"Scaler expects: {scaler.n_features_in_} features")
print(f"Selected features: {len(selected_features)} features")

# Import feature engineering
from app import create_statistical_features

# Test data
input_data = pd.DataFrame({
    'Driver Age': [30],
    'Driver Experience': [10],
    'Car Age': [5],
    'Previous Accidents': [0],
    'Annual Mileage (x1000 km)': [15.0]
})

# Create features
input_features = create_statistical_features(input_data, data_stats)
print(f"Created {len(input_features.columns)} features")

# Select features
input_features_selected = input_features[selected_features]
print(f"Selected features shape: {input_features_selected.shape}")

# Scale
input_scaled_full = scaler.transform(input_features_selected)
print(f"Scaled features shape: {input_scaled_full.shape}")

# Use first 19 features if we have 20
if input_scaled_full.shape[1] == 20:
    input_scaled = input_scaled_full[:, :19]
    print(f"Using first 19 features: {input_scaled.shape}")
else:
    input_scaled = input_scaled_full

# Predict
prediction = model.predict(input_scaled)[0]
print(f"Prediction successful! Premium: ${prediction:.2f}")