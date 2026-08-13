"""
ETA Prediction - Model Training
Reads: data/features/nyc_features.csv (output of src/features/build_features.py)
Trains and compares Random Forest and LightGBM regressors for trip_duration prediction.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb


# ---------------------------------------------------------------------------
# 1. Load engineered features
# ---------------------------------------------------------------------------
file_path = os.path.join('data', 'features', 'nyc_features.csv')
df = pd.read_csv(file_path, parse_dates=['pickup_datetime'])

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(df.columns.tolist())
print(df.info())


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
# Encode store_and_fwd_flag as binary (Y/N -> 1/0)
if 'store_and_fwd_flag' in df.columns:
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].apply(lambda x: 1 if x == 'Y' else 0)

# Drop identifier / raw timestamp columns not usable as model features.
# pickup_datetime was already decomposed into hour/day_of_week/hour_sin/etc.
# during feature engineering, so the raw timestamp itself isn't needed here.
drop_cols = [c for c in ['id', 'pickup_datetime'] if c in df.columns]
df = df.drop(columns=drop_cols)

print("\nColumns after preprocessing:")
print(df.columns.tolist())


# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
TARGET = 'trip_duration'
assert TARGET in df.columns, f"Expected target column '{TARGET}' not found in dataset"

y = df[TARGET]
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Quick missing-value check
train_na = X_train.isnull().sum()
test_na = X_test.isnull().sum()
if train_na.any() or test_na.any():
    print("\nMissing values found:")
    print("X_train:\n", train_na[train_na > 0])
    print("X_test:\n", test_na[test_na > 0])
else:
    print("\nNo missing values in X_train / X_test.")


# ---------------------------------------------------------------------------
# 4. Random Forest baseline
# ---------------------------------------------------------------------------
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, n_jobs=1, max_depth=15)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor Performance:")
print(f"  MAE:  {mae_rf:.2f}")
print(f"  RMSE: {rmse_rf:.2f}")
print(f"  R2:   {r2_rf:.2f}")


# ---------------------------------------------------------------------------
# 5. LightGBM
# ---------------------------------------------------------------------------
print("\nTraining LightGBM Regressor...")
lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)

print("LightGBM Regressor Performance:")
print(f"  MAE:  {mae_lgb:.2f}")
print(f"  RMSE: {rmse_lgb:.2f}")
print(f"  R2:   {r2_lgb:.2f}")


# ---------------------------------------------------------------------------
# 6. Compare models
# ---------------------------------------------------------------------------
performance_df = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM'],
    'MAE': [mae_rf, mae_lgb],
    'RMSE': [rmse_rf, rmse_lgb],
    'R2': [r2_rf, r2_lgb],
})
print("\nModel comparison:")
print(performance_df)

best_model_name = performance_df.loc[performance_df['MAE'].idxmin(), 'Model']
y_pred_best = y_pred_rf if best_model_name == 'Random Forest' else y_pred_lgb
print(f"\nBest model by MAE: {best_model_name}")

os.makedirs('reports', exist_ok=True)
performance_df.to_csv('reports/model_comparison.csv', index=False)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Trip Duration')
plt.ylabel('Predicted Trip Duration')
plt.title(f'{best_model_name}: Actual vs. Predicted Trip Duration')
plt.grid(True)
plt.tight_layout()
plt.savefig('reports/actual_vs_predicted.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 7. Feature importance (LightGBM)
# ---------------------------------------------------------------------------
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_model.feature_importances_,
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('LightGBM Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('reports/feature_importances.png', dpi=150)
plt.close()

print("\nTop 10 features:")
print(feature_importances.head(10))


# ---------------------------------------------------------------------------
# 8. Save best model
# ---------------------------------------------------------------------------
import joblib

os.makedirs('models', exist_ok=True)
best_model = rf_model if best_model_name == 'Random Forest' else lgb_model
model_path = os.path.join('models', 'best_model.pkl')
joblib.dump(best_model, model_path)
print(f"\nSaved best model ({best_model_name}) to {model_path}")