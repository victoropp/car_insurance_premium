# Project Structure - Car Insurance Premium Analytics Platform
## Videbimus AI

### 📁 Directory Organization

```
car_insurance_premium/
│
├── 📊 dashboard_updated.py     # Main dashboard application
├── 🏃 run_dashboard.py         # Dashboard launcher script
│
├── 📂 data/                    # Data files
│   ├── insurance_tranining_dataset.csv
│   ├── insurance_tranining_dataset_test.csv
│   ├── predictions_holdout_test.csv
│   ├── final_test_results.csv
│   ├── model_results.csv
│   └── feature_importance.csv
│
├── 📂 models/                  # Trained ML models
│   ├── stacking_linear.pkl
│   ├── stacking_ridge.pkl
│   └── voting_ensemble.pkl
│
├── 📂 src/                     # Source code modules
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── ensemble_models.py
│   ├── advanced_training_pipeline.py
│   ├── final_evaluation.py
│   ├── visualizations.py
│   ├── visualizations_updated.py
│   ├── dashboard.py            # Original dashboard
│   ├── simple_test_dashboard.py
│   ├── world_class_dashboard.py
│   └── worldclass_dashboard_complete.py
│
├── 📂 docs/                    # Documentation
│   ├── DASHBOARD_README.md
│   ├── PROFESSIONAL_DASHBOARD_README.md
│   ├── RESULTS_SUMMARY.md
│   └── best_hyperparameters.txt
│
├── 📂 assets/                  # Static assets (if any)
├── 📂 static/                  # Static files for web
├── 📂 tests/                   # Test files
│
├── 📋 requirements.txt         # Python dependencies
├── 🚀 Procfile                 # Heroku/Render deployment
├── 🔧 render.yaml              # Render.com configuration
├── 📖 README.md               # Project documentation
├── 🚫 .gitignore              # Git ignore rules
└── 📄 PROJECT_STRUCTURE.md    # This file

### 🌐 Deployment Information

**Website:** https://www.videbimusai.com  
**Email:** consulting@videbimusai.com  
**Developer:** Victor Collins Oppon  
**Company:** Videbimus AI

### 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   python dashboard_updated.py
   ```
   Or use the launcher:
   ```bash
   python run_dashboard.py
   ```

3. Access at: http://127.0.0.1:8050

### 📊 Model Performance

- **Best Model:** Stacking (Linear)
- **Test R² Score:** 0.9978
- **RMSE:** 0.272
- **MAE:** 0.201

### 🛠️ Technology Stack

- **Framework:** Dash/Plotly
- **ML Libraries:** Scikit-learn, XGBoost, CatBoost, LightGBM
- **Data Processing:** Pandas, NumPy
- **Deployment:** Render.com / Heroku
- **UI Components:** Dash Bootstrap Components