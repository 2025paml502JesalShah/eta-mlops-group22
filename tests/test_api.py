from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_predict():
    payload = {
        "vendor_id": 1,
        "pickup_datetime": "2016-06-15T08:30:00",
        "passenger_count": 2,
        "pickup_longitude": -73.9855,
        "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.9851,
        "dropoff_latitude": 40.7484,
        "store_and_fwd_flag": "N",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_duration_seconds" in data
    assert "predicted_duration_minutes" in data

    assert data["predicted_duration_seconds"] >= 0
    assert data["predicted_duration_minutes"] >= 0