# ETA MLOps Group 22

## NYC Taxi Trip Duration Prediction using MLOps

This project implements an end-to-end Machine Learning Operations (MLOps) workflow for predicting NYC taxi trip durations.

The project focuses on reproducible machine learning using Git, GitHub, DVC, automated feature engineering, model training, model evaluation, and experiment tracking.

---

## Team Members

| Student ID | Name |
|------------|------|
| 2025PAML589 | Sunidhi Sharma |
| 2025PAML502 | Jesal Shah |
| 2025PAML609 | Saakshi Arora |

---

## Project Objectives

- Predict NYC taxi trip duration using machine learning.
- Build a reproducible end-to-end MLOps pipeline.
- Version datasets and pipeline artifacts using DVC.
- Version source code and project configuration using Git and GitHub.
- Store large trained model files using Git LFS.
- Automate feature engineering and model training using DVC.
- Compare multiple machine learning models.
- Evaluate model performance using MAE, RMSE, and R².
- Track experiments, parameters, metrics, and models using MLflow.
- Deploy the trained model through a FastAPI prediction service.
- Containerize the API and MLflow services using Docker and Docker Compose.
- Monitor prediction performance and feature drift.
- Generate retraining decisions based on operational thresholds.
- Run automated project tests using GitHub Actions CI.
- Maintain reproducibility across the project.

---

## Dataset

Dataset: **NYC Taxi Trip Duration**

The project uses the NYC Taxi Trip Duration dataset.

The main datasets are versioned using DVC:

```text
data/raw/NYC.csv
data/processed/NYC_clean.csv
```

Large datasets and DVC pipeline artifacts are not stored directly in GitHub. They are versioned and managed using DVC with a Google Drive remote.

---

## Technology Stack

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- FastAPI
- Uvicorn
- MLflow
- Git
- GitHub
- Git LFS
- DVC
- Google Drive DVC Remote
- Docker
- Docker Compose
- PyYAML
- Pytest
- GitHub Actions

---

# Project Structure

```text
eta-mlops-group22/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── docs/
├── models/
│   ├── best_model.pkl
│   └── location_kmeans.pkl
│
├── reports/
│   ├── cleaning_report.txt
│   ├── feature_engineering_report.txt
│   ├── validation_report.txt
│   ├── training_summary.txt
│   ├── feature_importances.csv
│   ├── feature_importances.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.csv
│   ├── normal_trip_model_comparison.csv
│   ├── prediction_log.csv
│   ├── drift_report.txt
│   └── monitoring_report.txt
│
├── scripts/
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   └── clean.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   └── train_model.py
│   │
│   ├── monitoring/
│   │   ├── detect_drift.py
│   │   ├── log_predictions.py
│   │   └── monitor.py
│   │
│   └── utils/
│
├── tests/
│   ├── test_api.py
│   └── test_project.py
│
├── .dockerignore
├── .gitignore
├── .gitattributes
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

---

# Project Setup

## 1. Clone the Repository

```bash
git clone https://github.com/2025paml502JesalShah/eta-mlops-group22.git
cd eta-mlops-group22
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# DVC Setup

The project uses DVC for versioning large datasets, generated feature data, pipeline outputs, and model metrics.

The configured DVC remote is Google Drive.

Check the configured remote:

```bash
dvc remote list
```

The project uses a shared Google Drive DVC remote for storing large project artifacts.

Large trained model files are stored using Git LFS because the model artifacts are too large for normal Git storage.

---

## Download DVC Data

After cloning the repository, retrieve the versioned data and model artifacts using:

```bash
dvc pull
```

---

# Data Pipeline

The project follows the following pipeline:

```text
Raw Dataset
    │
    ▼
data/raw/NYC.csv
    │
    ▼
Data Cleaning
    │
    ▼
data/processed/NYC_clean.csv
    │
    ▼
Feature Engineering
    │
    ▼
data/features/nyc_features.csv
    │
    ▼
Model Training
    │
    ├── Random Forest Baseline
    ├── Random Forest Filtered
    └── LightGBM
    │
    ▼
Model Evaluation
    │
    ▼
models/best_model.pkl
```

The pipeline is defined in:

```text
dvc.yaml
```

The exact pipeline state is tracked in:

```text
dvc.lock
```

---

# Feature Engineering

Feature engineering is implemented in:

```text
src/features/build_features.py
```

Input:

```text
data/processed/NYC_clean.csv
```

Output:

```text
data/features/nyc_features.csv
```

The generated feature dataset contains approximately **1.45 million rows and 28 columns**.

---

## Time Features

The following time-based features are generated:

- `hour`
- `day_of_week`
- `day_of_month`
- `month`
- `is_weekend`
- `is_morning_rush`
- `is_evening_rush`
- `is_late_night`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `is_holiday`

---

## Distance and Trip Features

The following trip-related features are generated:

- `haversine_km`
- `manhattan_km`
- `bearing`

---

## Location Features

Pickup and dropoff locations are converted into clusters:

- `pickup_cluster`
- `dropoff_cluster`

The clustering configuration is:

```yaml
features:
  n_clusters: 100
  random_state: 42
```

MiniBatchKMeans is used for location clustering.

---

# Model Training

Model training is implemented in:

```text
src/models/train_model.py
```

The training pipeline performs:

1. Loading the feature dataset.
2. Preparing the input features.
3. Removing non-feature columns.
4. Splitting the dataset into training and testing sets.
5. Training a baseline Random Forest model.
6. Applying trip-duration outlier treatment.
7. Training an outlier-filtered Random Forest model.
8. Evaluating performance on normal trips.
9. Training a LightGBM comparison model.
10. Comparing model performance.
11. Selecting the best model.
12. Saving the final model.

---

# Model Parameters

The current model configuration is stored in:

```text
params.yaml
```

Current configuration:

```yaml
model:
  test_size: 0.2
  random_state: 42
  outlier_threshold_seconds: 21600
  n_estimators: 50
  max_depth: 15
```

The outlier threshold is:

```text
21,600 seconds = 6 hours
```

Random Forest uses:

```text
n_estimators = 50
max_depth = 15
random_state = 42
n_jobs = -1
```

---

# Outlier Treatment

Trip durations above 6 hours are treated as outliers during training.

```text
Outlier threshold: 21,600 seconds
```

Training data before filtering:

```text
1,161,859 rows
```

Training data after filtering:

```text
1,160,222 rows
```

Rows removed:

```text
1,637
```

Removal percentage:

```text
0.1409%
```

The test dataset is not filtered during the normal model evaluation.

---

# Model Results

Three models were compared:

1. Random Forest Baseline
2. Random Forest Filtered
3. LightGBM

## Overall Test Performance

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| Random Forest Baseline | 394.92 | 3184.83 | -0.0140 |
| Random Forest Filtered | **304.12** | **3110.20** | **0.0330** |
| LightGBM | 407.50 | 3303.78 | -0.0911 |

The **Random Forest Filtered** model achieved the lowest overall MAE and was selected as the final model.

---

# Normal Trip Evaluation

Normal trips are defined as trips with duration less than or equal to 6 hours.

Number of normal test trips:

```text
290,061 / 290,465
```

## Normal Trip Results

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| Random Forest Baseline | 280.99 | 783.75 | -0.4385 |
| Random Forest Filtered | **189.77** | **308.24** | **0.7775** |

The outlier-filtered Random Forest performs substantially better for normal trips, achieving an R² of 0.7775 and a MAE of 189.77 seconds.

---

# Best Model

The selected final model is:

```text
Random Forest Filtered
```

Saved to:

```text
models/best_model.pkl
```

The model artifact is tracked using Git LFS.

---

# Feature Importance

The most important features identified by the final Random Forest model include:

| Feature | Importance |
|--------|-----------:|
| haversine_km | 0.744128 |
| hour_cos | 0.041805 |
| bearing | 0.028457 |
| dropoff_latitude | 0.027419 |
| hour | 0.024435 |
| dropoff_longitude | 0.022752 |
| pickup_longitude | 0.018814 |
| pickup_latitude | 0.012839 |
| hour_sin | 0.011835 |
| day_of_week | 0.009749 |

The results show that trip distance, represented primarily by `haversine_km`, is the strongest predictor of trip duration in the current model.

---
# MLflow Experiment Tracking

MLflow is used to track model training experiments, parameters, metrics, and model artifacts.

The MLflow tracking server is available locally at:

```text
http://localhost:5000
```

The project uses the experiment:

```text
NYC Taxi ETA
```

Three models are evaluated and tracked:

- Random Forest Baseline
- Random Forest Filtered
- LightGBM

For each training run, MLflow records:

- Model parameters
- MAE
- RMSE
- R²
- Model artifacts
- Source training script

## Latest Model Results

The final training run produced:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest Baseline | 394.92 | 3184.83 | -0.0140 |
| Random Forest Filtered | **304.12** | **3110.20** | **0.0330** |
| LightGBM | 407.50 | 3303.78 | -0.0911 |

The Random Forest Filtered model achieved the lowest overall MAE and was selected as the final model.

## Model Registration

The selected model is registered in MLflow as:

```text
NYC Taxi ETA Model
```

The model artifact is also stored locally as:

```text
models/best_model.pkl
```

Large model files are tracked using Git LFS.

---
# FastAPI Prediction Service

The trained Random Forest Filtered model is exposed through a FastAPI prediction service.

The API runs on port `8000`.

## API Endpoints

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "Random Forest Filtered"
}
```

### Prediction

```text
POST /predict
```

The prediction endpoint accepts taxi trip information including:

- Vendor ID
- Pickup datetime
- Passenger count
- Pickup coordinates
- Dropoff coordinates
- Store-and-forward flag

Example response:

```json
{
  "predicted_duration_seconds": 1352.99,
  "predicted_duration_minutes": 22.55
}
```

## Interactive API Documentation

FastAPI automatically provides Swagger documentation at:

```text
http://localhost:8000/docs
```

The `/health` and `/predict` endpoints were tested successfully with HTTP 200 responses.

# Docker Deployment

The application is containerized using Docker.

The project includes:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Docker Compose runs two services:

| Service | Container | Port |
|---|---|---:|
| FastAPI API | `nyc-taxi-api` | 8000 |
| MLflow | `nyc-taxi-mlflow` | 5000 |

Start both services:

```bash
docker compose up -d
```

Check the running containers:

```bash
docker compose ps
```

Expected services:

```text
nyc-taxi-api
nyc-taxi-mlflow
```

The API is available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

MLflow:

```text
http://localhost:5000
```

Stop the services:

```bash
docker compose down
```
# Model Monitoring and Drift Detection

The project includes a monitoring layer to track prediction performance and detect changes in production feature distributions.

The monitoring components are located in:

```text
src/monitoring/
├── log_predictions.py
├── detect_drift.py
└── monitor.py
```

## Prediction Monitoring

Predictions are logged to:

```text
reports/prediction_log.csv
```

The prediction log contains:

- Trip ID
- Pickup datetime
- Actual trip duration
- Predicted duration
- Prediction error
- Absolute error

A total of 1,000 predictions were monitored during the evaluation.

The observed mean absolute error was:

```text
MAE = 441.44 seconds
```

The configured operational MAE threshold is:

```text
600 seconds
```

Therefore, model performance remained within the configured performance threshold.

## Feature Drift Detection

Population Stability Index (PSI) is used to detect changes in monitored feature distributions.

The configured PSI thresholds are:

| PSI | Interpretation |
|---:|---|
| < 0.10 | No significant drift |
| 0.10 - 0.25 | Moderate drift |
| > 0.25 | Significant drift |

The drift monitoring evaluated the following features:

| Feature | PSI | Status |
|---|---:|---|
| haversine_km | 0.0614 | No Drift |
| passenger_count | 0.0607 | No Drift |
| hour | **0.2522** | **Significant Drift** |
| day_of_week | 0.0018 | No Drift |

The `hour` feature exceeded the significant drift threshold.

## Retraining Trigger

The monitoring system uses two operational conditions to determine whether retraining is required:

1. MAE > 600 seconds
2. Any monitored feature PSI > 0.25

If either condition is satisfied, the system recommends retraining.

For the simulated production monitoring run:

```text
Drift status: SIGNIFICANT DRIFT DETECTED

DECISION: RETRAINING REQUIRED
```

When the retraining trigger fires, the DVC pipeline can be executed to regenerate features and retrain the model. The resulting model should then be evaluated against the existing production model before deployment.

# Continuous Integration

GitHub Actions is used to automatically test the project whenever changes are pushed to the `main` branch or a pull request is created against `main`.

The workflow is located at:

```text
.github/workflows/ci.yml
```

The CI pipeline performs the following steps:

1. Checks out the repository.
2. Downloads Git LFS model files.
3. Sets up Python 3.12.
4. Installs project dependencies.
5. Verifies the model files.
6. Runs the project test suite using Pytest.

The project currently contains:

```text
10 automated tests
```

All tests pass successfully:

```text
10 passed
```

The GitHub Actions workflow provides an automated quality check before changes are accepted into the project.

# Large Model Artifact Management

The trained model artifacts are larger than the standard GitHub file-size limit for normal Git storage.

Git LFS is therefore used to store the large model files:

```text
models/best_model.pkl
models/location_kmeans.pkl
```

The project also tracks the Jupyter notebook using Git LFS:

```text
Group_project.ipynb
```

The model files remain available through the GitHub repository while avoiding storage of large binary files directly in normal Git history.

# Reports

The project generates several reports:

```text
reports/
├── cleaning_report.txt
├── feature_engineering_report.txt
├── validation_report.txt
├── training_summary.txt
├── feature_importances.csv
├── feature_importances.png
└── actual_vs_predicted.png
```
Additional MLops repots include:

```text
reports/
├── model_comparison.csv
├── normal_trip_model_comparison.csv
├── prediction_log.csv
├── drift_report.txt
└── monitoring_report.txt

---

# Testing
The project includes automated tests for both the API and project components.

Run the complete test suite using:

```text
pytest

The current test suite contains:
```text
10 tests

Latest local test result:
```text
10 passed

A warning related to the Starlette/httpx TestClient dependency was reported, but all tests completed successfully.

---

# Reproducing the Pipeline

The complete pipeline can be reproduced using:

```bash
dvc repro
```

DVC automatically determines which stages need to be executed.

The current pipeline contains:

```text
build_features
      │
      ▼
train_model
```

Check pipeline status:

```bash
dvc status
```

Expected result when everything is up to date:

```text
Data and pipelines are up to date.
```

---

# DVC Pipeline

The DVC pipeline is defined in:

```text
dvc.yaml
```

The pipeline contains two main stages:

### 1. Build Features

```text
Input:
data/processed/NYC_clean.csv

Script:
src/features/build_features.py

Output:
data/features/nyc_features.csv
```

### 2. Train Model

```text
Input:
data/features/nyc_features.csv

Script:
src/models/train_model.py

Output:
models/best_model.pkl

Metrics:
reports/model_comparison.csv
reports/normal_trip_model_comparison.csv
```

Pipeline dependencies, parameters, hashes, and outputs are recorded in:

```text
dvc.lock
```

---

# Version Control Workflow

Git is used to version source code, configuration, documentation, and project metadata.

Typical workflow:

```bash
git status
git add .
git commit -m "Describe changes"
git push origin main
```

DVC is used for large files:

```bash
dvc repro
dvc push
```

After changing the DVC pipeline:

```bash
git add dvc.yaml dvc.lock params.yaml
git commit -m "Update DVC pipeline"
git push origin main
```

---

# Current Project Status

The following components have been completed:

- Git repository configured
- GitHub repository connected
- DVC configured
- Google Drive DVC remote configured
- Raw dataset versioned with DVC
- Processed dataset versioned with DVC
- Feature engineering implemented
- Feature dataset generated
- Feature dataset versioned with DVC
- DVC pipeline implemented
- DVC pipeline lock file generated
- Model training implemented
- Random Forest baseline implemented
- Outlier-filtered Random Forest implemented
- LightGBM comparison implemented
- Model evaluation implemented
- Model comparison implemented
- Feature importance analysis implemented
- Best model selection implemented
- MLflow experiment tracking implemented
- MLflow model registration implemented
- FastAPI prediction service implemented
- Docker deployment implemented
- Docker Compose configuration implemented
- Prediction logging implemented
- PSI-based drift detection implemented
- Retraining trigger implemented
- Automated Pytest test suite implemented
- GitHub Actions CI implemented
- Large model files tracked using Git LFS
- DVC pipeline successfully reproduced
- DVC artifacts successfully pushed to remote storage
- Git changes successfully pushed to GitHub

---

# Reproduce the Current Project

After cloning the repository:

```bash
git clone https://github.com/2025paml502JesalShah/eta-mlops-group22.git
cd eta-mlops-group22
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Retrieve DVC files:

```bash
dvc pull
```

Reproduce the pipeline:

```bash
dvc repro
```

Check pipeline status:

```bash
dvc status
```
Run tests:

```bash
pytest
```
---
# Running the Application

## Start Docker Services

Start the FastAPI and MLflow services using Docker Compose:

```bash
docker compose up -d
```

Check the running services:

```bash
docker compose ps
```

## Access FastAPI

API:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

## Access MLflow

MLflow tracking server:

```text
http://localhost:5000
```

## Stop Services

Stop the Docker Compose services:

```bash
docker compose down
```
---

# Future Work

Potential improvements to the project include:

- Hyperparameter tuning and systematic model optimization.
- Additional machine learning model experimentation.
- More advanced drift detection techniques.
- Fully automated model retraining and validation workflows.
- Production deployment to a cloud environment.
- Real-time monitoring dashboards.
- Integration with a production-scale data ingestion pipeline.
- Automated model promotion and rollback strategies.

---

# Conclusion

This project implements an end-to-end MLOps pipeline for NYC taxi trip duration prediction.

The completed workflow covers:

- Data cleaning and validation.
- Feature engineering.
- Reproducible data and model pipelines using DVC.
- Machine learning model comparison.
- Outlier treatment.
- Experiment tracking using MLflow.
- Model registration and artifact management.
- FastAPI model serving.
- Docker and Docker Compose deployment.
- Prediction logging and model monitoring.
- PSI-based feature drift detection.
- Operational retraining triggers.
- Automated testing with Pytest.
- Continuous integration using GitHub Actions.
- Large model artifact management using Git LFS.

The final selected model is the outlier-filtered Random Forest, which achieved the best overall MAE among the evaluated models and strong performance on normal trips.

The project demonstrates how a machine learning model can be developed, tracked, deployed, monitored, and maintained as a reproducible MLOps workflow.

---

# License

Academic project for BITS Pilani Post Graduate Certificate (AI & ML).