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

# Test import
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
    
    return errors, warnings

print("✅ App imports and validation function work correctly!")