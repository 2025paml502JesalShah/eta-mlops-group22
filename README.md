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
- Build a reproducible MLOps pipeline.
- Track datasets and large model artifacts using DVC.
- Version source code and project configuration using Git and GitHub.
- Automate feature engineering and model training using DVC.
- Compare multiple machine learning models.
- Evaluate model performance using MAE, RMSE, and R².
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

Large datasets are not stored directly in GitHub. They are managed through DVC.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Git
- GitHub
- DVC
- Google Drive DVC Remote
- PyYAML

---

# Project Structure

```text
eta-mlops-group22/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── docs/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── features/
│   │   └── build_features.py
│   │
│   └── models/
│       └── train_model.py
│
├── tests/
│
├── .dvc/
├── .gitignore
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

The project uses DVC for versioning large datasets, generated features, model artifacts, and model metrics.

The configured DVC remote is Google Drive.

Check the configured remote:

```bash
dvc remote list
```

The project uses a shared Google Drive DVC remote for storing large project artifacts.

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
| Random Forest Baseline | 394.03 | 3156.03 | 0.0043 |
| Random Forest Filtered | **304.36** | **3110.23** | 0.0330 |
| LightGBM | 406.30 | 3260.86 | -0.0630 |

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
| Random Forest Baseline | 280.05 | 651.61 | 0.0056 |
| Random Forest Filtered | **190.00** | **308.53** | **0.7771** |

The outlier-filtered Random Forest performs substantially better for normal trips.

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

The model is tracked using DVC rather than Git because model files can be large.

---

# Feature Importance

The most important features identified by the final Random Forest model include:

| Feature | Importance |
|--------|-----------:|
| haversine_km | 0.743538 |
| hour_cos | 0.042530 |
| bearing | 0.028633 |
| dropoff_latitude | 0.027544 |
| hour | 0.023390 |
| dropoff_longitude | 0.023041 |
| pickup_longitude | 0.018842 |
| pickup_latitude | 0.012963 |
| hour_sin | 0.012137 |
| day_of_week | 0.009841 |

The results show that trip distance, represented primarily by `haversine_km`, is the strongest predictor of trip duration in the current model.

---

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

Model comparison metrics are tracked through DVC:

```text
reports/model_comparison.csv
reports/normal_trip_model_comparison.csv
```

These files are excluded from Git tracking and managed by DVC.

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
- Model artifact versioned using DVC
- Model metrics tracked using DVC
- Pipeline successfully reproduced using `dvc repro`
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

---

# Future Work

The next stages of the project can include:

- Hyperparameter tuning
- Additional model experimentation
- Improved feature engineering
- Model serving API
- Docker containerization
- CI/CD automation
- Automated model testing
- Model monitoring
- Deployment
- Prediction interface

---

# License

Academic project for BITS Pilani Post Graduate Certificate (AI & ML).