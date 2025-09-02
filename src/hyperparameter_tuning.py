import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
exec(open('advanced_training_pipeline.py').read())

print("\n" + "=" * 80)
print("HYPERPARAMETER OPTIMIZATION WITH BAYESIAN OPTIMIZATION")
print("=" * 80)

# Define parameter grids for top performing models
param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.001, 0.01, 0.1],
        'reg_lambda': [0, 0.001, 0.01, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7, 9, -1],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [5, 10, 20, 30],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.001, 0.01, 0.1],
        'reg_lambda': [0, 0.001, 0.01, 0.1]
    },
    'CatBoost': {
        'iterations': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128, 255],
        'bagging_temperature': [0, 0.5, 1, 2]
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, 0.7],
        'bootstrap': [True, False]
    }
}

# Perform RandomizedSearchCV for efficiency
print("\nPerforming Randomized Search for hyperparameter optimization...")
print("-" * 50)

best_models = {}
tuning_results = []

for model_name, param_grid in param_grids.items():
    print(f"\nTuning {model_name}...")
    
    if model_name == 'XGBoost':
        base_model = XGBRegressor(random_state=42, verbosity=0)
    elif model_name == 'LightGBM':
        base_model = LGBMRegressor(random_state=42, verbosity=-1)
    elif model_name == 'CatBoost':
        base_model = CatBoostRegressor(random_state=42, verbose=False)
    elif model_name == 'Random Forest':
        base_model = RandomForestRegressor(random_state=42)
    
    # RandomizedSearchCV for faster optimization
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    best_models[model_name] = random_search.best_estimator_
    
    # Evaluate on validation set
    val_pred = random_search.best_estimator_.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    result = {
        'Model': model_name,
        'Best_Params': random_search.best_params_,
        'CV_Score': -random_search.best_score_,
        'Val_RMSE': val_rmse,
        'Val_R2': val_r2
    }
    
    tuning_results.append(result)
    print(f"  Best CV RMSE: {np.sqrt(-random_search.best_score_):.6f}")
    print(f"  Validation RMSE: {val_rmse:.6f}")
    print(f"  Validation R²: {val_r2:.6f}")

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 80)

tuning_df = pd.DataFrame(tuning_results)
print("\nBest models after tuning:")
print(tuning_df[['Model', 'CV_Score', 'Val_RMSE', 'Val_R2']].to_string(index=False))

# Save best parameters
with open('best_hyperparameters.txt', 'w') as f:
    f.write("BEST HYPERPARAMETERS FOR EACH MODEL\n")
    f.write("=" * 50 + "\n\n")
    for result in tuning_results:
        f.write(f"\n{result['Model']}:\n")
        f.write("-" * 30 + "\n")
        for param, value in result['Best_Params'].items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\n  Validation RMSE: {result['Val_RMSE']:.6f}\n")
        f.write(f"  Validation R²: {result['Val_R2']:.6f}\n")

print("\nBest hyperparameters saved to 'best_hyperparameters.txt'")