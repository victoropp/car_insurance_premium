import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED CAR INSURANCE PREMIUM PREDICTION PIPELINE")
print("=" * 80)

# Load data
df = pd.read_csv('insurance_tranining_dataset.csv')
df_test = pd.read_csv('insurance_tranining_dataset_test.csv')

print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Test dataset loaded: {df_test.shape[0]} samples")

# Feature Engineering
print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

def create_features(df):
    df_feat = df.copy()
    
    # Interaction features
    df_feat['Age_Experience_Ratio'] = df_feat['Driver Age'] / (df_feat['Driver Experience'] + 1)
    df_feat['Accidents_Per_Year_Driving'] = df_feat['Previous Accidents'] / (df_feat['Driver Experience'] + 1)
    df_feat['Mileage_Per_Year_Driving'] = df_feat['Annual Mileage (x1000 km)'] / (df_feat['Driver Experience'] + 1)
    df_feat['Car_Age_Driver_Age_Ratio'] = df_feat['Car Age'] / df_feat['Driver Age']
    df_feat['Experience_Rate'] = df_feat['Driver Experience'] / df_feat['Driver Age']
    
    # Polynomial features for key variables
    df_feat['Driver_Age_Squared'] = df_feat['Driver Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Driver Experience'] ** 2
    df_feat['Accidents_Squared'] = df_feat['Previous Accidents'] ** 2
    
    # Risk indicators
    df_feat['High_Risk_Driver'] = ((df_feat['Driver Age'] < 25) | (df_feat['Driver Age'] > 65)).astype(int)
    df_feat['New_Driver'] = (df_feat['Driver Experience'] < 2).astype(int)
    df_feat['Old_Car'] = (df_feat['Car Age'] > 10).astype(int)
    df_feat['High_Mileage'] = (df_feat['Annual Mileage (x1000 km)'] > df_feat['Annual Mileage (x1000 km)'].median()).astype(int)
    
    # Composite risk score
    df_feat['Risk_Score'] = (
        df_feat['High_Risk_Driver'] * 2 + 
        df_feat['New_Driver'] * 3 + 
        df_feat['Previous Accidents'] * 4 + 
        df_feat['Old_Car'] * 1 +
        df_feat['High_Mileage'] * 1
    )
    
    return df_feat

# Apply feature engineering
df_engineered = create_features(df)
df_test_engineered = create_features(df_test)

print(f"Features after engineering: {df_engineered.shape[1]}")
print("New features created:")
new_features = [col for col in df_engineered.columns if col not in df.columns]
for feat in new_features:
    if feat != 'Insurance Premium ($)':
        print(f"  - {feat}")

# Prepare data for modeling
X = df_engineered.drop('Insurance Premium ($)', axis=1)
y = df_engineered['Insurance Premium ($)']
X_test_final = df_test_engineered.drop('Insurance Premium ($)', axis=1) if 'Insurance Premium ($)' in df_test_engineered.columns else df_test_engineered

# Three-way split: Train (60%), Validation (20%), Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nData Split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_test_final_scaled = scaler.transform(X_test_final)

print("\nData preprocessing completed with RobustScaler")