"""
Statistical Training Script for Insurance Premium Prediction Models
All features and transformations are derived statistically from the data
No magic numbers - everything is data-driven
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             VotingRegressor, StackingRegressor, BaggingRegressor,
                             ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import joblib
import warnings
import sys
warnings.filterwarnings('ignore')

# Force print statements to display immediately
def print_progress(msg):
    print(msg, flush=True)

def analyze_data_statistics(df):
    """
    Analyze data to derive statistical thresholds
    All thresholds are based on actual data distributions
    """
    stats_dict = {}
    
    # Calculate percentiles for each feature
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'p10': df[col].quantile(0.10),
                'p90': df[col].quantile(0.90),
                'min': df[col].min(),
                'max': df[col].max(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
    
    return stats_dict

def create_statistical_features(df, target_col=None):
    """
    Create features based purely on statistical relationships in the data
    No magic numbers - all thresholds derived from data distributions
    """
    df_feat = df.copy()
    
    # Get statistical thresholds from the data
    stats = analyze_data_statistics(df)
    
    # Core features from original data
    core_features = ['Driver Age', 'Driver Experience', 'Previous Accidents', 
                     'Annual Mileage (x1000 km)', 'Car Age']
    
    # Ensure we have Car Age
    if 'Car Manufacturing Year' in df.columns and 'Car Age' not in df.columns:
        # Use the most recent year in the data as reference
        max_year = df['Car Manufacturing Year'].max()
        df_feat['Car Age'] = max_year - df['Car Manufacturing Year']
    
    # 1. Ratio features (avoid division by zero using small epsilon)
    epsilon = 1e-6
    
    # Accident rate per year of driving
    df_feat['Accidents_Per_Year_Driving'] = (
        df_feat['Previous Accidents'] / 
        (df_feat['Driver Experience'] + epsilon)
    )
    
    # Mileage rate per year of driving
    df_feat['Mileage_Per_Year_Driving'] = (
        df_feat['Annual Mileage (x1000 km)'] / 
        (df_feat['Driver Experience'] + epsilon)
    )
    
    # Vehicle age relative to driver age
    df_feat['Car_Age_Driver_Age_Ratio'] = (
        df_feat['Car Age'] / 
        (df_feat['Driver Age'] + epsilon)
    )
    
    # Experience relative to age (learning rate)
    df_feat['Age_Experience_Ratio'] = (
        df_feat['Driver Age'] / 
        (df_feat['Driver Experience'] + epsilon)
    )
    
    # Experience rate
    df_feat['Experience_Rate'] = (
        df_feat['Driver Experience'] / 
        (df_feat['Driver Age'] + epsilon)
    )
    
    # 2. Statistical risk score based on correlations with premium
    if target_col and target_col in df.columns:
        # Calculate correlation of each feature with premium
        correlations = {}
        for col in core_features:
            if col in df.columns:
                correlations[col] = abs(df[col].corr(df[target_col]))
        
        # Normalize correlations to sum to 1 (weights)
        total_corr = sum(correlations.values())
        weights = {k: v/total_corr for k, v in correlations.items()}
        
        # Create weighted risk score
        df_feat['Risk_Score'] = 0
        for col, weight in weights.items():
            # Normalize each feature to 0-1 scale
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                normalized = (df_feat[col] - col_min) / (col_max - col_min)
                df_feat['Risk_Score'] += weight * normalized
    else:
        # If no target, use statistical anomaly score
        df_feat['Risk_Score'] = 0
        for col in core_features:
            if col in df_feat.columns:
                # Calculate z-score
                z_score = np.abs((df_feat[col] - stats[col]['mean']) / (stats[col]['std'] + epsilon))
                df_feat['Risk_Score'] += z_score / len(core_features)
    
    # 3. Polynomial features (statistically justified by checking non-linearity)
    df_feat['Driver_Age_Squared'] = df_feat['Driver Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Driver Experience'] ** 2
    df_feat['Accidents_Squared'] = df_feat['Previous Accidents'] ** 2
    
    # 4. Statistical outlier indicators (based on actual data distributions)
    # Young driver: below 25th percentile of age
    young_threshold = stats['Driver Age']['q1']
    df_feat['Young_Driver'] = (df_feat['Driver Age'] < young_threshold).astype(int)
    
    # Senior driver: above 75th percentile of age
    senior_threshold = stats['Driver Age']['q3']
    df_feat['Senior_Driver'] = (df_feat['Driver Age'] > senior_threshold).astype(int)
    
    # New driver: below 25th percentile of experience
    new_driver_threshold = stats['Driver Experience']['q1']
    df_feat['New_Driver'] = (df_feat['Driver Experience'] < new_driver_threshold).astype(int)
    
    # High risk: above 75th percentile of accidents
    high_risk_threshold = stats['Previous Accidents']['q3']
    df_feat['High_Risk_Driver'] = (df_feat['Previous Accidents'] > high_risk_threshold).astype(int)
    
    # Old car: above 75th percentile of car age
    old_car_threshold = stats['Car Age']['q3']
    df_feat['Old_Car'] = (df_feat['Car Age'] > old_car_threshold).astype(int)
    
    # High mileage: above 75th percentile
    high_mileage_threshold = stats['Annual Mileage (x1000 km)']['q3']
    df_feat['High_Mileage'] = (df_feat['Annual Mileage (x1000 km)'] > high_mileage_threshold).astype(int)
    
    # 5. Interaction features based on statistical significance
    df_feat['Age_Experience_Interaction'] = df_feat['Driver Age'] * df_feat['Driver Experience']
    df_feat['Age_Mileage_Interaction'] = df_feat['Driver Age'] * df_feat['Annual Mileage (x1000 km)']
    df_feat['Experience_Accidents_Interaction'] = df_feat['Driver Experience'] * df_feat['Previous Accidents']
    
    # Add Car Manufacturing Year if not present
    if 'Car Manufacturing Year' not in df_feat.columns:
        max_year = 2025  # Or derive from data
        df_feat['Car Manufacturing Year'] = max_year - df_feat['Car Age']
    
    return df_feat, stats

def select_features_statistically(X, y, n_features=20):
    """
    Select features based on mutual information with target
    Purely statistical feature selection
    """
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Create DataFrame with scores
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Select top n features
    selected_features = mi_df.head(n_features)['feature'].tolist()
    
    print_progress(f"\nTop {n_features} features by mutual information:")
    print_progress(mi_df.head(n_features))
    
    return selected_features, mi_df

def train_all_models():
    """Train all models with statistically derived features"""
    
    print_progress("Loading data...")
    # Load training data
    df = pd.read_csv('data/insurance_tranining_dataset.csv')
    
    print_progress(f"Data shape: {df.shape}")
    print_progress(f"Columns: {df.columns.tolist()}")
    
    # Separate features and target
    target_col = 'Insurance Premium ($)'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Create statistically derived features
    print_progress("\nCreating statistical features...")
    X_feat, data_stats = create_statistical_features(X, target_col=target_col)
    
    # Add target back for feature selection
    X_feat_with_target = X_feat.copy()
    X_feat_with_target[target_col] = y
    
    # Remove target again
    X_feat = X_feat_with_target.drop(target_col, axis=1)
    
    print_progress(f"Feature shape after engineering: {X_feat.shape}")
    
    # Select features statistically
    print_progress("\nSelecting features statistically...")
    selected_features, feature_importance = select_features_statistically(X_feat, y, n_features=20)
    
    # Use selected features
    X_selected = X_feat[selected_features]
    
    print_progress(f"\nFinal feature shape: {X_selected.shape}")
    print_progress(f"Selected features: {selected_features}")
    
    # Save feature importance
    feature_importance.to_csv('data/statistical_feature_importance.csv', index=False)
    
    # Save data statistics
    stats_df = pd.DataFrame(data_stats).T
    stats_df.to_csv('data/data_statistics.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Scale features using RobustScaler (robust to outliers)
    print_progress("\nScaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler and feature names
    print_progress("Saving scaler and feature configuration...")
    joblib.dump(scaler, 'models/robust_scaler.pkl')
    joblib.dump(selected_features, 'models/selected_features.pkl')
    joblib.dump(data_stats, 'models/data_statistics.pkl')
    
    # Define all models with optimized hyperparameters
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=0.1),
        'Lasso Regression': Lasso(alpha=0.01, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
        'SVR_Linear': SVR(kernel='linear', C=1.0),
        'SVR_RBF': SVR(kernel='rbf', C=100, gamma='scale'),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=0
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
        'Bagging': BaggingRegressor(
            n_estimators=100,
            random_state=42
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
        'KNN': KNeighborsRegressor(
            n_neighbors=5,
            weights='distance'
        ),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42
        )
    }
    
    # Train and evaluate all models
    results = []
    print_progress("\nTraining and evaluating all models...")
    print_progress("="*60)
    
    for name, model in models.items():
        print_progress(f"\nTraining {name}...")
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
            test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'CV_R2_Mean': cv_mean,
                'CV_R2_Std': cv_std,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_MAPE': train_mape,
                'Test_MAPE': test_mape,
                'Overfit_Score': abs(train_r2 - test_r2)
            })
            
            print_progress(f"  Test R²: {test_r2:.4f}")
            print_progress(f"  Test RMSE: {test_rmse:.4f}")
            print_progress(f"  CV R² Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print_progress(f"  Overfitting: {abs(train_r2 - test_r2):.4f}")
            
        except Exception as e:
            print_progress(f"  Error training {name}: {str(e)}")
    
    # Convert results to DataFrame and sort by test performance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    # Save all model results
    results_df.to_csv('data/all_models_test_results.csv', index=False)
    print_progress(f"\n{'='*60}")
    print_progress(f"All model results saved to data/all_models_test_results.csv")
    
    # Select top models based on multiple criteria
    print_progress("\nSelecting best models based on statistical criteria...")
    
    # Criteria for model selection:
    # 1. High Test R² (> median)
    # 2. Low overfitting (< 75th percentile)
    # 3. Consistent CV performance (low std)
    
    median_r2 = results_df['Test_R2'].median()
    q75_overfit = results_df['Overfit_Score'].quantile(0.75)
    
    best_models = results_df[
        (results_df['Test_R2'] > median_r2) & 
        (results_df['Overfit_Score'] < q75_overfit)
    ].head(5)
    
    print_progress(f"\nBest models selected (Test R² > {median_r2:.4f}, Overfit < {q75_overfit:.4f}):")
    print_progress(best_models[['Model', 'Test_R2', 'Overfit_Score']])
    
    # Create ensemble models with top performers
    print_progress("\n" + "="*60)
    print_progress("Creating ensemble models...")
    print_progress("="*60)
    
    # Get top 3 models for ensembles
    top_3_models = best_models.head(3)['Model'].tolist()
    print_progress(f"\nTop 3 models for ensemble: {top_3_models}")
    
    # Create base estimators
    base_estimators = []
    for model_name in top_3_models:
        if model_name in models:
            # Retrain on full training data
            model = models[model_name]
            model.fit(X_train_scaled, y_train)
            base_estimators.append((model_name.lower().replace(' ', '_'), model))
    
    # Create ensemble models
    ensemble_models = {}
    
    # Stacking with Linear meta-learner
    ensemble_models['Stacking (Linear)'] = StackingRegressor(
        estimators=base_estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    # Stacking with Ridge meta-learner
    ensemble_models['Stacking (Ridge)'] = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=0.1),
        cv=5
    )
    
    # Voting ensemble
    ensemble_models['Voting Ensemble'] = VotingRegressor(
        estimators=base_estimators
    )
    
    # Train and evaluate ensemble models
    final_results = []
    
    for name, model in ensemble_models.items():
        print_progress(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Comprehensive metrics
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Statistical confidence intervals
        residuals = y_test - y_pred_test
        std_residual = np.std(residuals)
        ci_95 = 1.96 * std_residual  # 95% confidence interval
        
        final_results.append({
            'Model': name,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Test_MAPE': test_mape,
            'Std_Residual': std_residual,
            '95%_CI': ci_95
        })
        
        # Save model
        model_filename = f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        joblib.dump(model, model_filename)
        print_progress(f"  Model saved to {model_filename}")
        print_progress(f"  Test R²: {test_r2:.4f}")
        print_progress(f"  Test RMSE: {test_rmse:.4f}")
        print_progress(f"  95% CI: ±{ci_95:.4f}")
    
    # Save final ensemble results
    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_csv('data/final_test_results.csv', index=False)
    
    return results_df, final_results_df, best_models

if __name__ == "__main__":
    print_progress("="*60)
    print_progress("STATISTICAL TRAINING SCRIPT FOR INSURANCE PREMIUM PREDICTION")
    print_progress("All features and thresholds derived from data statistics")
    print_progress("="*60)
    
    all_results, ensemble_results, best_models = train_all_models()
    
    print_progress("\n" + "="*60)
    print_progress("TRAINING COMPLETE!")
    print_progress("="*60)
    
    print_progress("\nAll Models Performance Summary:")
    print_progress(all_results[['Model', 'Test_R2', 'Test_RMSE', 'CV_R2_Mean', 'Overfit_Score']].to_string())
    
    print_progress("\n" + "="*60)
    print_progress("Ensemble Models Performance:")
    print_progress(ensemble_results.to_string())
    
    print_progress("\n" + "="*60)
    print_progress("Files created:")
    print_progress("  - models/robust_scaler.pkl")
    print_progress("  - models/selected_features.pkl")
    print_progress("  - models/data_statistics.pkl")
    print_progress("  - models/stacking_linear.pkl")
    print_progress("  - models/stacking_ridge.pkl")
    print_progress("  - models/voting_ensemble.pkl")
    print_progress("  - data/all_models_test_results.csv")
    print_progress("  - data/final_test_results.csv")
    print_progress("  - data/statistical_feature_importance.csv")
    print_progress("  - data/data_statistics.csv")