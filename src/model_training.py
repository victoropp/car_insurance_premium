import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing from the pipeline
exec(open('advanced_training_pipeline.py').read())

print("\n" + "=" * 80)
print("TRAINING ADVANCED MACHINE LEARNING MODELS")
print("=" * 80)

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)
    
    # Training predictions
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    train_mape = mean_absolute_percentage_error(y_train, train_pred)
    
    # Validation predictions
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    val_mape = mean_absolute_percentage_error(y_val, val_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results = {
        'Model': model_name,
        'Train_RMSE': train_rmse,
        'Train_MAE': train_mae,
        'Train_R2': train_r2,
        'Train_MAPE': train_mape,
        'Val_RMSE': val_rmse,
        'Val_MAE': val_mae,
        'Val_R2': val_r2,
        'Val_MAPE': val_mape,
        'CV_RMSE': cv_rmse,
        'Overfit_Score': abs(train_rmse - val_rmse)
    }
    
    return model, results

# Initialize models
models = {
    # Linear Models
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    
    # Tree-based Models
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42),
    'Bagging': BaggingRegressor(n_estimators=100, random_state=42),
    
    # Advanced Boosting
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbosity=-1),
    'CatBoost': CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, random_state=42, verbose=False),
    
    # Other Models
    'KNN': KNeighborsRegressor(n_neighbors=10),
    'SVR_RBF': SVR(kernel='rbf', C=100, gamma=0.001),
    'SVR_Linear': SVR(kernel='linear', C=100),
    
    # Neural Network
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', 
                                  solver='adam', max_iter=1000, random_state=42)
}

# Train all models
results_list = []
trained_models = {}

print("\nTraining models...")
print("-" * 50)

for name, model in models.items():
    print(f"Training {name}...")
    trained_model, results = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, name)
    results_list.append(results)
    trained_models[name] = trained_model

# Create results DataFrame
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('Val_RMSE')

print("\n" + "=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)
print("\nTop 10 Models by Validation RMSE:")
print(results_df[['Model', 'Train_RMSE', 'Val_RMSE', 'Val_R2', 'Val_MAPE', 'CV_RMSE', 'Overfit_Score']].head(10).to_string(index=False))

# Save results
results_df.to_csv('model_results.csv', index=False)
print("\nDetailed results saved to 'model_results.csv'")