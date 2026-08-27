from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import holidays
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
KMEANS_PATH = BASE_DIR / "models" / "location_kmeans.pkl"


# ============================================================
# Load trained models
# ============================================================

model = joblib.load(MODEL_PATH)
kmeans = joblib.load(KMEANS_PATH)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="NYC Taxi ETA Prediction API",
    description="Predict NYC taxi trip duration using the trained Random Forest model.",
    version="1.0.0",
)


# ============================================================
# Request schema
# ============================================================

class TripRequest(BaseModel):
    vendor_id: int
    pickup_datetime: datetime
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str = "N"


# ============================================================
# Haversine distance
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ============================================================
# Feature engineering
# ============================================================

def create_features(request: TripRequest):

    # --------------------------------------------------------
    # Date/time features
    # --------------------------------------------------------

    dt = pd.Timestamp(request.pickup_datetime)

    hour = dt.hour
    day_of_week = dt.dayofweek
    day_of_month = dt.day
    month = dt.month

    is_weekend = int(day_of_week in [5, 6])

    is_morning_rush = int(7 <= hour <= 10)
    is_evening_rush = int(16 <= hour <= 19)
    is_late_night = int(hour >= 23 or hour <= 5)

    # --------------------------------------------------------
    # Cyclical time features
    # --------------------------------------------------------

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    # --------------------------------------------------------
    # Location values
    # --------------------------------------------------------

    pickup_lat = request.pickup_latitude
    pickup_lon = request.pickup_longitude

    dropoff_lat = request.dropoff_latitude
    dropoff_lon = request.dropoff_longitude

    # --------------------------------------------------------
    # Haversine distance
    # --------------------------------------------------------

    haversine_km = haversine(
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
    )

    # --------------------------------------------------------
    # Manhattan distance
    # --------------------------------------------------------

    manhattan_km = (
        haversine(
            pickup_lat,
            pickup_lon,
            dropoff_lat,
            pickup_lon,
        )
        + haversine(
            dropoff_lat,
            pickup_lon,
            dropoff_lat,
            dropoff_lon,
        )
    )

    # --------------------------------------------------------
    # Bearing
    # --------------------------------------------------------

    bearing = np.degrees(
        np.arctan2(
            np.sin(np.radians(dropoff_lon - pickup_lon))
            * np.cos(np.radians(dropoff_lat)),
            np.cos(np.radians(pickup_lat))
            * np.sin(np.radians(dropoff_lat))
            - np.sin(np.radians(pickup_lat))
            * np.cos(np.radians(dropoff_lat))
            * np.cos(np.radians(dropoff_lon - pickup_lon)),
        )
    )

    # --------------------------------------------------------
    # KMeans location clusters
    # --------------------------------------------------------

    pickup_cluster = kmeans.predict(
        [[pickup_lat, pickup_lon]]
    )[0]

    dropoff_cluster = kmeans.predict(
        [[dropoff_lat, dropoff_lon]]
    )[0]

    # --------------------------------------------------------
    # Convert store_and_fwd_flag
    #
    # API receives:
    #     "N" or "Y"
    #
    # Model receives:
    #     0 or 1
    # --------------------------------------------------------

    store_and_fwd_flag = (
        1
        if request.store_and_fwd_flag.upper() == "Y"
        else 0
    )

    # --------------------------------------------------------
    # Create feature DataFrame
    # --------------------------------------------------------

    features = pd.DataFrame(
        [
            {
                "vendor_id": request.vendor_id,
                "passenger_count": request.passenger_count,
                "pickup_longitude": pickup_lon,
                "pickup_latitude": pickup_lat,
                "dropoff_longitude": dropoff_lon,
                "dropoff_latitude": dropoff_lat,

                # IMPORTANT: numeric value for ML model
                "store_and_fwd_flag": (
    1 if request.store_and_fwd_flag.upper() == "Y" else 0
),

                "hour": hour,
                "day_of_week": day_of_week,
                "day_of_month": day_of_month,
                "month": month,
                "is_weekend": is_weekend,
                "is_morning_rush": is_morning_rush,
                "is_evening_rush": is_evening_rush,
                "is_late_night": is_late_night,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "is_holiday": int(
                    dt.date() in holidays.US(years=[dt.year])
                ),
                "haversine_km": haversine_km,
                "manhattan_km": manhattan_km,
                "bearing": bearing,
                "pickup_cluster": pickup_cluster,
                "dropoff_cluster": dropoff_cluster,
            }
        ]
    )

    return features


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "NYC Taxi ETA Prediction API",
        "status": "running",
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest Filtered",
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(request: TripRequest):

    # Create model features
    features = create_features(request)

    # Make prediction
    prediction_seconds = float(
        model.predict(features)[0]
    )

    # Prevent negative predictions
    prediction_seconds = max(
        0,
        prediction_seconds,
    )

    # Convert seconds to minutes
    prediction_minutes = prediction_seconds / 60

    return {
        "predicted_duration_seconds": round(
            prediction_seconds,
            2,
        ),
        "predicted_duration_minutes": round(
            prediction_minutes,
            2,
        ),
    }