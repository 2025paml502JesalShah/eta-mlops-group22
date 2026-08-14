"""
MLflow experiment tracking for the ETA prediction project.

This script logs:
- Model parameters
- Model comparison metrics
- Normal-trip evaluation metrics
- Best model
- Trained Random Forest model artifact
"""

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = "models/best_model.pkl"
MODEL_COMPARISON_PATH = "reports/model_comparison.csv"
NORMAL_TRIP_COMPARISON_PATH = "reports/normal_trip_model_comparison.csv"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "NYC Taxi Trip Duration Prediction"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("MLflow Experiment Tracking")
    print("=" * 70)

    # Use local MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create/select experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load model comparison results
    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)

    # Load normal-trip comparison results
    normal_df = pd.read_csv(NORMAL_TRIP_COMPARISON_PATH)

    # Identify best model by MAE
    best_row = comparison_df.loc[comparison_df["MAE"].idxmin()]
    best_model_name = best_row["Model"]

    print(f"Best model: {best_model_name}")

    # Start MLflow run
    with mlflow.start_run(run_name="Random Forest Filtered"):

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("model_variant", "Filtered")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("outlier_threshold_seconds", 21600)
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 15)

        # -------------------------------------------------
        # Metrics - Overall evaluation
        # -------------------------------------------------

        filtered_row = comparison_df[
            comparison_df["Model"] == "Random Forest Filtered"
        ].iloc[0]

        mlflow.log_metric("mae", float(filtered_row["MAE"]))
        mlflow.log_metric("rmse", float(filtered_row["RMSE"]))
        mlflow.log_metric("r2", float(filtered_row["R2"]))

        # -------------------------------------------------
        # Metrics - Normal trip evaluation
        # -------------------------------------------------

        normal_filtered = normal_df[
            normal_df["Model"] == "Random Forest Filtered - Normal Trips"
        ]

        if not normal_filtered.empty:
            normal_row = normal_filtered.iloc[0]

            mlflow.log_metric(
                "normal_trip_mae",
                float(normal_row["MAE"])
            )

            mlflow.log_metric(
                "normal_trip_rmse",
                float(normal_row["RMSE"])
            )

            mlflow.log_metric(
                "normal_trip_r2",
                float(normal_row["R2"])
            )

        # -------------------------------------------------
        # Log reports
        # -------------------------------------------------

        if os.path.exists(MODEL_COMPARISON_PATH):
            mlflow.log_artifact(MODEL_COMPARISON_PATH)

        if os.path.exists(NORMAL_TRIP_COMPARISON_PATH):
            mlflow.log_artifact(NORMAL_TRIP_COMPARISON_PATH)

        if os.path.exists("reports/training_summary.txt"):
            mlflow.log_artifact("reports/training_summary.txt")

        if os.path.exists("reports/feature_importances.csv"):
            mlflow.log_artifact("reports/feature_importances.csv")

        # -------------------------------------------------
        # Log trained model
        # -------------------------------------------------

        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)

            mlflow.sklearn.log_model(
                model,
                name="best_model"
            )

            print("Model logged successfully.")

        # -------------------------------------------------
        # Tags
        # -------------------------------------------------

        mlflow.set_tag("project", "ETA MLOps Group 22")
        mlflow.set_tag("dataset", "NYC Taxi Trip Duration")
        mlflow.set_tag("best_model", best_model_name)

        print()
        print("MLflow run completed successfully.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()