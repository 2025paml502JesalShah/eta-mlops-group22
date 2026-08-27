from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "features" / "nyc_features.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = REPORT_DIR / "prediction_log.csv"


FEATURE_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_late_night",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "haversine_km",
    "manhattan_km",
    "bearing",
    "pickup_cluster",
    "dropoff_cluster",
]


def main():

    print("Loading data and trained model...")

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    print(f"Loaded {len(df):,} rows.")
    print(f"Loaded model: {type(model).__name__}")

    # --------------------------------------------------------
    # Select reproducible monitoring sample
    # --------------------------------------------------------

    sample = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    ).copy()

    # --------------------------------------------------------
    # Prepare model features
    # --------------------------------------------------------

    X = sample[FEATURE_COLUMNS].copy()

    # train_model.py converts Y/N into 1/0
    X["store_and_fwd_flag"] = (
        X["store_and_fwd_flag"]
        .map({"Y": 1, "N": 0})
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Generate REAL predictions
    # --------------------------------------------------------

    predictions = model.predict(X)

    sample["predicted_duration"] = predictions

    # --------------------------------------------------------
    # Calculate monitoring errors
    # --------------------------------------------------------

    sample["prediction_error"] = (
        sample["trip_duration"]
        - sample["predicted_duration"]
    )

    sample["absolute_error"] = (
        sample["prediction_error"].abs()
    )

    # --------------------------------------------------------
    # Calculate MAE and RMSE
    # --------------------------------------------------------

    mae = sample["absolute_error"].mean()

    rmse = np.sqrt(
        (
            sample["prediction_error"] ** 2
        ).mean()
    )

    # --------------------------------------------------------
    # Save prediction log
    # --------------------------------------------------------

    prediction_log = sample[
        [
            "id",
            "pickup_datetime",
            "trip_duration",
            "predicted_duration",
            "prediction_error",
            "absolute_error",
        ]
    ].copy()

    prediction_log.to_csv(
        LOG_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("")
    print("NYC TAXI ETA - PREDICTION MONITORING")
    print("=" * 55)

    print(f"Logged predictions: {len(prediction_log):,}")
    print(f"MAE: {mae:.2f} seconds")
    print(f"RMSE: {rmse:.2f} seconds")

    print("")
    print("Sample predictions:")
    print(
        prediction_log.head(5).to_string(
            index=False
        )
    )

    print("")
    print(f"Saved prediction log to: {LOG_PATH}")


if __name__ == "__main__":
    main()