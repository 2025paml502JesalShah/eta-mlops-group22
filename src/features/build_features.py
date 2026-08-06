import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import holidays

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def add_time_features(df):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_of_month'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['is_morning_rush'] = df['hour'].between(7, 10).astype(int)
    df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)
    df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    years = df['pickup_datetime'].dt.year.unique().tolist()
    us_hol = holidays.US(years=years)
    df['is_holiday'] = df['pickup_datetime'].dt.date.astype('datetime64[ns]').isin(us_hol).astype(int)
    return df

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

def add_distance_features(df):
    df['haversine_km'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['manhattan_km'] = (
        haversine(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['pickup_longitude']) +
        haversine(df['dropoff_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    )
    df['bearing'] = np.degrees(np.arctan2(
        np.sin(np.radians(df['dropoff_longitude'] - df['pickup_longitude'])) * np.cos(np.radians(df['dropoff_latitude'])),
        np.cos(np.radians(df['pickup_latitude'])) * np.sin(np.radians(df['dropoff_latitude'])) -
        np.sin(np.radians(df['pickup_latitude'])) * np.cos(np.radians(df['dropoff_latitude'])) *
        np.cos(np.radians(df['dropoff_longitude'] - df['pickup_longitude']))
    ))
    return df

def add_location_clusters(df, n_clusters=100, random_state=42):
    coords = np.vstack([
        df[['pickup_latitude', 'pickup_longitude']].values,
        df[['dropoff_latitude', 'dropoff_longitude']].values
    ])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                              n_init=10, batch_size=10000).fit(coords)
    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
    return df, kmeans

def main():
    params = load_params()
    fe_params = params.get('features', {})

    input_path = Path("data/processed/NYC_clean.csv")
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=['pickup_datetime', 'dropoff_datetime'])

    df = add_time_features(df)
    df = add_distance_features(df)
    df, kmeans_model = add_location_clusters(df, n_clusters=fe_params.get('n_clusters', 100))

    # Drop leakage-prone / raw columns not needed downstream
    df = df.drop(columns=['dropoff_datetime'])

    output_path = output_dir / "nyc_features.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns to {output_path}")

if __name__ == "__main__":
    main()