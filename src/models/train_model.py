"""
ETA Prediction - Model Training

Reads:
    data/features/nyc_features.csv

Trains:
    1. Baseline Random Forest
    2. Outlier-filtered Random Forest
    3. LightGBM comparison model

The final model is selected using MAE and saved to:
    models/best_model.pkl

Model parameters are read from params.yaml.
"""

import os
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Load parameters
# ---------------------------------------------------------------------------

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

model_params = params.get("model", {})

TEST_SIZE = model_params.get("test_size", 0.2)
RANDOM_STATE = model_params.get("random_state", 42)
OUTLIER_THRESHOLD = model_params.get(
    "outlier_threshold_seconds",
    21600
)
N_ESTIMATORS = model_params.get("n_estimators", 50)
MAX_DEPTH = model_params.get("max_depth", 15)

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("NYC Taxi ETA")

print("Model parameters:")
print(f"  Outlier threshold: {OUTLIER_THRESHOLD:,} seconds")
print(f"  Outlier threshold: {OUTLIER_THRESHOLD / 3600:.1f} hours")
print(f"  Random state: {RANDOM_STATE}")
print(f"  Test size: {TEST_SIZE}")
print(f"  Number of trees: {N_ESTIMATORS}")
print(f"  Max depth: {MAX_DEPTH}")


# ---------------------------------------------------------------------------
# 2. Load engineered features
# ---------------------------------------------------------------------------

file_path = Path("data/features/nyc_features.csv")

df = pd.read_csv(
    file_path,
    parse_dates=["pickup_datetime"]
)

print(f"\nLoaded {df.shape[0]} rows, {df.shape[1]} columns")
print(df.columns.tolist())


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

# Encode store_and_fwd_flag as binary.
if "store_and_fwd_flag" in df.columns:
    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"]
        .map({"Y": 1, "N": 0})
        .fillna(0)
        .astype(int)
    )

# Drop identifier and raw timestamp.
drop_cols = [
    col for col in ["id", "pickup_datetime"]
    if col in df.columns
]

df = df.drop(columns=drop_cols)

print("\nColumns after preprocessing:")
print(df.columns.tolist())


# ---------------------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------------------

TARGET = "trip_duration"

if TARGET not in df.columns:
    raise ValueError(
        f"Expected target column '{TARGET}' not found."
    )

y = df[TARGET]
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTrain/test split:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test:  {y_test.shape}")


# ---------------------------------------------------------------------------
# 5. Missing-value check
# ---------------------------------------------------------------------------

train_missing = X_train.isnull().sum()
test_missing = X_test.isnull().sum()

if train_missing.any() or test_missing.any():
    print("\nMissing values found:")
    print(train_missing[train_missing > 0])
    print(test_missing[test_missing > 0])
    raise ValueError("Missing values detected in model features.")

print("\nNo missing values in X_train / X_test.")


# ---------------------------------------------------------------------------
# 6. Helper function for evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.4f}")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ---------------------------------------------------------------------------
# 7. Baseline Random Forest
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Training baseline Random Forest...")
print("=" * 70)

rf_baseline = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

with mlflow.start_run(run_name="Random Forest Baseline"):

    rf_baseline.fit(X_train, y_train)

    y_pred_rf_baseline = rf_baseline.predict(X_test)

    baseline_results = evaluate_model(
        "Random Forest Baseline",
        y_test,
        y_pred_rf_baseline
    )

    mlflow.log_params({
        "model": "RandomForestRegressor",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "outlier_threshold": OUTLIER_THRESHOLD,
        "training_rows": len(X_train),
    })

    mlflow.log_metrics({
        "MAE": baseline_results["MAE"],
        "RMSE": baseline_results["RMSE"],
        "R2": baseline_results["R2"],
    })

    mlflow.sklearn.log_model(
        rf_baseline,
        "model"
    )


# ---------------------------------------------------------------------------
# 8. Outlier-filtered Random Forest
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Applying duration outlier treatment...")
print("=" * 70)

train_mask = y_train <= OUTLIER_THRESHOLD

X_train_filtered = X_train.loc[train_mask]
y_train_filtered = y_train.loc[train_mask]

removed_rows = len(y_train) - len(y_train_filtered)
removed_pct = removed_rows / len(y_train) * 100

print(f"Original training rows: {len(y_train):,}")
print(f"Filtered training rows: {len(y_train_filtered):,}")
print(f"Removed training rows: {removed_rows:,}")
print(f"Removal percentage: {removed_pct:.4f}%")


print("\nTraining outlier-filtered Random Forest...")

rf_filtered = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

with mlflow.start_run(run_name="Random Forest Filtered"):

    rf_filtered.fit(
        X_train_filtered,
        y_train_filtered
    )

    y_pred_rf_filtered = rf_filtered.predict(X_test)

    filtered_results = evaluate_model(
        "Random Forest Filtered",
        y_test,
        y_pred_rf_filtered
    )

    mlflow.log_params({
        "model": "RandomForestRegressor",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "outlier_threshold": OUTLIER_THRESHOLD,
        "training_rows": len(X_train_filtered),
        "removed_rows": removed_rows,
        "removed_percentage": removed_pct,
    })

    mlflow.log_metrics({
        "MAE": filtered_results["MAE"],
        "RMSE": filtered_results["RMSE"],
        "R2": filtered_results["R2"],
    })

    mlflow.sklearn.log_model(
        rf_filtered,
        "model"
    )


# ---------------------------------------------------------------------------
# 9. Evaluate normal trips separately
# ---------------------------------------------------------------------------

normal_test_mask = y_test <= OUTLIER_THRESHOLD

print("\n" + "=" * 70)
print("Normal-trip evaluation")
print("=" * 70)

print(
    f"Normal test rows: "
    f"{normal_test_mask.sum():,} / {len(y_test):,}"
)

baseline_normal_results = evaluate_model(
    "Random Forest Baseline - Normal Trips",
    y_test[normal_test_mask],
    y_pred_rf_baseline[normal_test_mask]
)

filtered_normal_results = evaluate_model(
    "Random Forest Filtered - Normal Trips",
    y_test[normal_test_mask],
    y_pred_rf_filtered[normal_test_mask]
)


# ---------------------------------------------------------------------------
# 10. LightGBM comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Training LightGBM comparison model...")
print("=" * 70)

lgb_model = lgb.LGBMRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

with mlflow.start_run(run_name="LightGBM"):

    lgb_model.fit(X_train, y_train)

    y_pred_lgb = lgb_model.predict(X_test)

    lgb_results = evaluate_model(
        "LightGBM",
        y_test,
        y_pred_lgb
    )

    mlflow.log_params({
        "model": "LightGBM",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": len(X_train),
    })

    mlflow.log_metrics({
        "MAE": lgb_results["MAE"],
        "RMSE": lgb_results["RMSE"],
        "R2": lgb_results["R2"],
    })

    mlflow.lightgbm.log_model(
        lgb_model,
        "model"
    )


# ---------------------------------------------------------------------------
# 11. Model comparison
# ---------------------------------------------------------------------------

comparison_df = pd.DataFrame([
    baseline_results,
    filtered_results,
    lgb_results,
])

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))


# Select best model based on MAE.
best_row = comparison_df.loc[
    comparison_df["MAE"].idxmin()
]

best_model_name = best_row["Model"]

if best_model_name == "Random Forest Baseline":
    best_model = rf_baseline
    best_predictions = y_pred_rf_baseline

elif best_model_name == "Random Forest Filtered":
    best_model = rf_filtered
    best_predictions = y_pred_rf_filtered

else:
    best_model = lgb_model
    best_predictions = y_pred_lgb

print(f"\nBest model by MAE: {best_model_name}")


# ---------------------------------------------------------------------------
# 12. Create reports directory
# ---------------------------------------------------------------------------

reports_dir = Path("reports")
models_dir = Path("models")

reports_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 13. Save model comparison report
# ---------------------------------------------------------------------------

comparison_df.to_csv(
    reports_dir / "model_comparison.csv",
    index=False
)


# ---------------------------------------------------------------------------
# 14. Save normal-trip evaluation report
# ---------------------------------------------------------------------------

normal_results_df = pd.DataFrame([
    baseline_normal_results,
    filtered_normal_results,
])

normal_results_df.to_csv(
    reports_dir / "normal_trip_model_comparison.csv",
    index=False
)


# ---------------------------------------------------------------------------
# 15. Actual vs predicted plot
# ---------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.3,
    s=5
)

min_value = min(
    y_test.min(),
    best_predictions.min()
)

max_value = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    linewidth=2
)

plt.xlabel("Actual Trip Duration")
plt.ylabel("Predicted Trip Duration")
plt.title(
    f"{best_model_name}: Actual vs Predicted Trip Duration"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    reports_dir / "actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------------------------
# 16. Feature importance
# ---------------------------------------------------------------------------

if hasattr(best_model, "feature_importances_"):

    feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False
    )

    feature_importances.to_csv(
        reports_dir / "feature_importances.csv",
        index=False
    )

    plt.figure(figsize=(12, 8))

    top_features = feature_importances.head(15)

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"{best_model_name} Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        reports_dir / "feature_importances.png",
        dpi=150
    )

    plt.close()

    print("\nTop 10 features:")
    print(
        feature_importances
        .head(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------------
# 17. Save final model
# ---------------------------------------------------------------------------

model_path = models_dir / "best_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\n" + "=" * 70)
print("FINAL MODEL")
print("=" * 70)
print(f"Selected model: {best_model_name}")
print(f"Saved to: {model_path}")


# ---------------------------------------------------------------------------
# 18. Save experiment summary
# ---------------------------------------------------------------------------

summary_path = reports_dir / "training_summary.txt"

with open(summary_path, "w") as f:

    f.write("ETA Prediction Model Training Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write("Model parameters\n")
    f.write("-" * 30 + "\n")
    f.write(
        f"Outlier threshold: "
        f"{OUTLIER_THRESHOLD} seconds "
        f"({OUTLIER_THRESHOLD / 3600:.1f} hours)\n"
    )
    f.write(f"Test size: {TEST_SIZE}\n")
    f.write(f"Random state: {RANDOM_STATE}\n")
    f.write(f"Number of trees: {N_ESTIMATORS}\n")
    f.write(f"Max depth: {MAX_DEPTH}\n\n")

    f.write("Outlier treatment\n")
    f.write("-" * 30 + "\n")
    f.write(f"Original training rows: {len(y_train):,}\n")
    f.write(
        f"Filtered training rows: "
        f"{len(y_train_filtered):,}\n"
    )
    f.write(f"Removed rows: {removed_rows:,}\n")
    f.write(f"Removal percentage: {removed_pct:.4f}%\n\n")

    f.write("Model comparison\n")
    f.write("-" * 30 + "\n")
    f.write(
        comparison_df.to_string(index=False)
    )
    f.write("\n\n")

    f.write("Normal-trip evaluation\n")
    f.write("-" * 30 + "\n")
    f.write(
        normal_results_df.to_string(index=False)
    )
    f.write("\n\n")

    f.write(
        f"Best model by MAE: "
        f"{best_model_name}\n"
    )


print("\nTraining pipeline completed successfully.")