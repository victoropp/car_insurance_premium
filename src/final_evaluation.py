import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
exec(open('advanced_training_pipeline.py').read())

print("\n" + "=" * 80)
print("FINAL MODEL EVALUATION ON TEST SET")
print("=" * 80)

# Load saved models
print("\nLoading saved models...")
voting_ensemble = joblib.load('voting_ensemble.pkl')
stacking_linear = joblib.load('stacking_linear.pkl')
stacking_ridge = joblib.load('stacking_ridge.pkl')

# Evaluate on test set
def final_evaluation(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    # Calculate prediction intervals
    residuals = y_test - predictions
    std_residual = np.std(residuals)
    
    return {
        'Model': model_name,
        'Test_RMSE': rmse,
        'Test_MAE': mae,
        'Test_R2': r2,
        'Test_MAPE': mape * 100,
        'Std_Residual': std_residual,
        '95%_CI': 1.96 * std_residual
    }

# Test all ensemble models
test_results = []

models_to_test = {
    'Voting Ensemble': voting_ensemble,
    'Stacking (Linear)': stacking_linear,
    'Stacking (Ridge)': stacking_ridge
}

print("\nEvaluating models on test set...")
print("-" * 50)

for name, model in models_to_test.items():
    result = final_evaluation(model, X_test_scaled, y_test, name)
    test_results.append(result)
    print(f"{name}: RMSE={result['Test_RMSE']:.4f}, R²={result['Test_R2']:.4f}")

# Create results DataFrame
test_results_df = pd.DataFrame(test_results)
test_results_df = test_results_df.sort_values('Test_RMSE')

print("\n" + "=" * 80)
print("FINAL TEST SET RESULTS")
print("=" * 80)
print(test_results_df.to_string(index=False))

# Select best model
best_model = stacking_linear  # Based on validation results
print(f"\nBest Model Selected: Stacking (Linear)")

# Make predictions on holdout test data
print("\n" + "=" * 80)
print("PREDICTIONS ON HOLDOUT TEST SET")
print("=" * 80)

holdout_predictions = best_model.predict(X_test_final_scaled)

# Create submission file
submission_df = pd.DataFrame({
    'Index': range(len(holdout_predictions)),
    'Predicted_Premium': holdout_predictions
})

submission_df.to_csv('predictions_holdout_test.csv', index=False)
print(f"\nPredictions saved to 'predictions_holdout_test.csv'")
print(f"Number of predictions: {len(holdout_predictions)}")
print(f"Prediction range: ${holdout_predictions.min():.2f} - ${holdout_predictions.max():.2f}")
print(f"Mean prediction: ${holdout_predictions.mean():.2f}")
print(f"Median prediction: ${np.median(holdout_predictions):.2f}")

# Feature importance analysis
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importance from base estimators
feature_names = X_train.columns.tolist()
importances = []

# Extract from XGBoost (one of the base estimators)
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_importance = xgb_model.feature_importances_

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Save all results
test_results_df.to_csv('final_test_results.csv', index=False)
feature_importance_df.to_csv('feature_importance.csv', index=False)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nFiles generated:")
print("  - model_results.csv: All model comparisons")
print("  - best_hyperparameters.txt: Optimal hyperparameters")
print("  - final_test_results.csv: Final test set evaluation")
print("  - predictions_holdout_test.csv: Predictions for holdout test set")
print("  - feature_importance.csv: Feature importance rankings")
print("  - *.pkl files: Saved model objects")