import os
import yaml
import pandas as pd


def test_params_yaml_exists():
    """Check that the project configuration exists."""
    assert os.path.exists("params.yaml")


def test_params_yaml_structure():
    """Check that required configuration sections exist."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    assert "dataset" in params
    assert "features" in params
    assert "model" in params

    assert "n_clusters" in params["features"]
    assert "random_state" in params["features"]

    assert "test_size" in params["model"]
    assert "outlier_threshold_seconds" in params["model"]
    assert "n_estimators" in params["model"]
    assert "max_depth" in params["model"]


def test_dvc_pipeline_exists():
    """Check that the DVC pipeline configuration exists."""
    assert os.path.exists("dvc.yaml")
    assert os.path.exists("dvc.lock")


def test_training_script_exists():
    """Check that the model training script exists."""
    assert os.path.exists("src/models/train_model.py")


def test_feature_engineering_script_exists():
    """Check that the feature engineering script exists."""
    assert os.path.exists("src/features/build_features.py")


def test_feature_dataset_exists():
    """Check that the generated feature dataset exists."""
    path = "data/features/nyc_features.csv"

    assert os.path.exists(path)

    # Read only a small sample so the test remains fast.
    df = pd.read_csv(path, nrows=10)

    expected_columns = [
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "trip_duration",
        "hour",
        "day_of_week",
        "haversine_km",
        "manhattan_km",
        "bearing",
        "pickup_cluster",
        "dropoff_cluster",
    ]

    for column in expected_columns:
        assert column in df.columns


def test_model_configuration():
    """Check that the current model configuration is valid."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model = params["model"]

    assert model["test_size"] > 0
    assert model["test_size"] < 1

    assert model["outlier_threshold_seconds"] > 0
    assert model["n_estimators"] > 0
    assert model["max_depth"] > 0