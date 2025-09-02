"""
Local Development Runner for Videbimus AI Insurance Premium Analytics
Streamlit Dashboard

This script provides an easy way to run the dashboard locally with
proper configuration and error handling.
"""

import streamlit as st
import subprocess
import sys
import os

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import pandas
        import numpy  
        import plotly
        import sklearn
        import joblib
        print("SUCCESS: All required packages are installed")
        return True
    except ImportError as e:
        print(f"ERROR: Missing required package: {str(e)}")
        print("Note: Please run: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if all required data files exist"""
    required_files = [
        'data/insurance_tranining_dataset.csv',
        'data/final_test_results.csv', 
        'data/model_results.csv',
        'data/feature_importance.csv',
        'models/stacking_linear.pkl',
        'models/stacking_ridge.pkl',
        'models/voting_ensemble.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("SUCCESS: All required data and model files found")
        return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("Starting Videbimus AI Insurance Premium Analytics Dashboard...")
    print("="*70)
    print("                      VIDEBIMUS AI")
    print("           Insurance Premium Analytics Platform")
    print("                   STREAMLIT VERSION")
    print("="*70)
    print()
    print("   Developed by: Victor Collins Oppon")
    print("   Company: Videbimus AI")
    print("   Website: https://www.videbimusai.com")
    print("   Contact: consulting@videbimusai.com")
    print()
    print("="*70)
    print()
    
    if not check_requirements():
        return
    
    if not check_data_files():
        print("Note: Please ensure all data and model files are in the correct directories")
        return
    
    print("Dashboard will open in your default browser")
    print("Local URL: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError running dashboard: {str(e)}")

if __name__ == "__main__":
    run_dashboard()