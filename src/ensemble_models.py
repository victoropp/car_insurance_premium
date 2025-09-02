import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing and best models
exec(open('advanced_training_pipeline.py').read())
exec(open('hyperparameter_tuning.py').read())

print("\n" + "=" * 80)
print("ADVANCED ENSEMBLE METHODS")
print("=" * 80)

# Create ensemble models
print("\nCreating ensemble models...")
print("-" * 50)

# 1. Voting Ensemble with weighted average
print("\n1. Weighted Voting Ensemble")
voting_regressor = VotingRegressor(
    estimators=[
        ('xgb', best_models['XGBoost']),
        ('lgb', best_models['LightGBM']),
        ('cat', best_models['CatBoost']),
        ('rf', best_models['Random Forest'])
    ],
    weights=[3, 2, 2, 1]  # Give more weight to better performing models
)

voting_regressor.fit(X_train_scaled, y_train)
voting_pred_val = voting_regressor.predict(X_val_scaled)
voting_rmse = np.sqrt(mean_squared_error(y_val, voting_pred_val))
voting_r2 = r2_score(y_val, voting_pred_val)

print(f"   Validation RMSE: {voting_rmse:.6f}")
print(f"   Validation R²: {voting_r2:.6f}")

# 2. Stacking with Linear Meta-learner
print("\n2. Stacking Ensemble (Linear Meta-learner)")
stacking_linear = StackingRegressor(
    estimators=[
        ('xgb', best_models['XGBoost']),
        ('lgb', best_models['LightGBM']),
        ('cat', best_models['CatBoost']),
        ('rf', best_models['Random Forest'])
    ],
    final_estimator=LinearRegression(),
    cv=5
)

stacking_linear.fit(X_train_scaled, y_train)
stacking_linear_pred = stacking_linear.predict(X_val_scaled)
stacking_linear_rmse = np.sqrt(mean_squared_error(y_val, stacking_linear_pred))
stacking_linear_r2 = r2_score(y_val, stacking_linear_pred)

print(f"   Validation RMSE: {stacking_linear_rmse:.6f}")
print(f"   Validation R²: {stacking_linear_r2:.6f}")

# 3. Stacking with Ridge Meta-learner
print("\n3. Stacking Ensemble (Ridge Meta-learner)")
stacking_ridge = StackingRegressor(
    estimators=[
        ('xgb', best_models['XGBoost']),
        ('lgb', best_models['LightGBM']),
        ('cat', best_models['CatBoost']),
        ('rf', best_models['Random Forest'])
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stacking_ridge.fit(X_train_scaled, y_train)
stacking_ridge_pred = stacking_ridge.predict(X_val_scaled)
stacking_ridge_rmse = np.sqrt(mean_squared_error(y_val, stacking_ridge_pred))
stacking_ridge_r2 = r2_score(y_val, stacking_ridge_pred)

print(f"   Validation RMSE: {stacking_ridge_rmse:.6f}")
print(f"   Validation R²: {stacking_ridge_r2:.6f}")

# 4. Advanced Blending
print("\n4. Custom Blending Ensemble")
# Get predictions from all models
predictions = {
    'XGBoost': best_models['XGBoost'].predict(X_val_scaled),
    'LightGBM': best_models['LightGBM'].predict(X_val_scaled),
    'CatBoost': best_models['CatBoost'].predict(X_val_scaled),
    'RandomForest': best_models['Random Forest'].predict(X_val_scaled)
}

# Optimize blending weights
from scipy.optimize import minimize

def blend_predictions(weights):
    blend = np.zeros_like(predictions['XGBoost'])
    for i, (name, pred) in enumerate(predictions.items()):
        blend += weights[i] * pred
    return blend

def blend_rmse(weights):
    blend = blend_predictions(weights)
    return np.sqrt(mean_squared_error(y_val, blend))

# Constraint: weights sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in range(len(predictions))]
initial_weights = [0.25] * len(predictions)

result = minimize(blend_rmse, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

blend_pred = blend_predictions(optimal_weights)
blend_rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
blend_r2 = r2_score(y_val, blend_pred)

print(f"   Optimal weights: {dict(zip(predictions.keys(), optimal_weights))}")
print(f"   Validation RMSE: {blend_rmse:.6f}")
print(f"   Validation R²: {blend_r2:.6f}")

# Compile ensemble results
ensemble_results = pd.DataFrame([
    {'Ensemble': 'Voting (Weighted)', 'Val_RMSE': voting_rmse, 'Val_R2': voting_r2},
    {'Ensemble': 'Stacking (Linear)', 'Val_RMSE': stacking_linear_rmse, 'Val_R2': stacking_linear_r2},
    {'Ensemble': 'Stacking (Ridge)', 'Val_RMSE': stacking_ridge_rmse, 'Val_R2': stacking_ridge_r2},
    {'Ensemble': 'Optimized Blending', 'Val_RMSE': blend_rmse, 'Val_R2': blend_r2}
])

print("\n" + "=" * 80)
print("ENSEMBLE MODELS COMPARISON")
print("=" * 80)
print(ensemble_results.to_string(index=False))

# Save best ensemble model
best_ensemble_idx = ensemble_results['Val_RMSE'].idxmin()
best_ensemble_name = ensemble_results.loc[best_ensemble_idx, 'Ensemble']

print(f"\nBest Ensemble Model: {best_ensemble_name}")
print(f"RMSE: {ensemble_results.loc[best_ensemble_idx, 'Val_RMSE']:.6f}")
print(f"R²: {ensemble_results.loc[best_ensemble_idx, 'Val_R2']:.6f}")

# Save models
joblib.dump(voting_regressor, 'voting_ensemble.pkl')
joblib.dump(stacking_linear, 'stacking_linear.pkl')
joblib.dump(stacking_ridge, 'stacking_ridge.pkl')
print("\nEnsemble models saved as .pkl files")