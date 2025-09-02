#!/usr/bin/env python
"""
Car Insurance Premium Analytics Dashboard Launcher

This script launches the interactive dashboard for analyzing car insurance premiums,
visualizing model performance, and testing predictions.

Usage:
    python run_dashboard.py

The dashboard will be available at http://127.0.0.1:8050
"""

import webbrowser
import time
import sys
import os

def main():
    print("\n" + "="*70)
    print("🚗 CAR INSURANCE PREMIUM ANALYTICS DASHBOARD")
    print("="*70)
    print("\n📊 Advanced Machine Learning Dashboard for Insurance Analysis")
    print("\n🚀 Starting dashboard server...")
    print("\n" + "-"*70)
    
    dashboard_url = "http://127.0.0.1:8050"
    
    print(f"\n✅ Dashboard Features:")
    print("   • Comprehensive Data Overview")
    print("   • Advanced Feature Analysis")
    print("   • Model Performance Comparison")
    print("   • Prediction Analysis")
    print("   • Risk Profiling")
    print("   • Interactive Model Testing Interface")
    
    print("\n" + "-"*70)
    print(f"\n🌐 Dashboard URL: {dashboard_url}")
    print("\n⏳ Opening browser in 3 seconds...")
    
    time.sleep(3)
    
    try:
        webbrowser.open(dashboard_url)
        print("\n✅ Browser opened successfully!")
    except:
        print(f"\n⚠️  Could not open browser automatically.")
        print(f"   Please open your browser and navigate to: {dashboard_url}")
    
    print("\n" + "-"*70)
    print("\n📝 Instructions:")
    print("   1. The dashboard is now running")
    print("   2. Use your browser to interact with the visualizations")
    print("   3. Test model predictions with custom inputs")
    print("   4. Press CTRL+C in this terminal to stop the server")
    print("\n" + "="*70 + "\n")
    
    try:
        from dashboard import app
        app.run(debug=False, port=8050, host='127.0.0.1')
    except ImportError:
        print("\n❌ Error: Could not import dashboard module.")
        print("   Please ensure dashboard.py and visualizations.py are in the current directory.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 Dashboard server stopped by user")
        print("="*70 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()