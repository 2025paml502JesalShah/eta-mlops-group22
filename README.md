# ETA MLOps Group 22

## NYC Taxi Trip Duration Prediction using MLOps

This project implements an end-to-end Machine Learning Operations (MLOps) workflow for predicting NYC taxi trip durations. The repository follows MLOps best practices, including version control with Git, data versioning with DVC, reproducible project structure, and collaborative development.

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
- Build a reproducible MLOps project.
- Track datasets using DVC.
- Maintain source code with Git and GitHub.
- Enable collaborative development.

---

## Dataset

Dataset: **NYC Taxi Trip Duration (Kaggle)**

Current datasets tracked using DVC:

- `data/raw/NYC.csv`
- `data/processed/NYC_clean.csv`

---

## Technology Stack

- Python 3.13
- Git
- GitHub
- DVC (Data Version Control)
- Pandas
- NumPy
- Scikit-learn

---

## Project Structure

```text
eta-mlops-group22/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
├── tests/
│
├── README.md
├── requirements.txt
├── params.yaml
├── .dvc/
└── .gitignore
```

---

## Project Setup

Clone the repository:

```bash
git clone https://github.com/2025paml502JesalShah/eta-mlops-group22.git
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download versioned datasets:

```bash
dvc pull
```

---

## Data Versioning

This project uses **DVC** with a shared Google Drive remote for dataset versioning.

Track a new dataset:

```bash
dvc add <dataset>
git add <dataset>.dvc
git commit -m "Track dataset"
git push
dvc push
```

Retrieve datasets:

```bash
dvc pull
```

---

## Feature Engineering

The project includes a reproducible feature-engineering pipeline that transforms the cleaned NYC taxi dataset into a model-ready feature dataset.

### Pipeline

```text
data/processed/NYC_clean.csv
        │
        ▼
src/features/build_features.py
        │
        ▼
data/features/nyc_features.csv
```

### Input

```text
data/processed/NYC_clean.csv
```

### Output

```text
data/features/nyc_features.csv
```

The generated feature dataset contains 28 columns and approximately 14.5 million rows.

### Engineered Features

#### Time Features

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

#### Distance and Trip Features

- `haversine_km`
- `manhattan_km`
- `bearing`

#### Location Features

- `pickup_cluster`
- `dropoff_cluster`

Location clusters are generated using MiniBatchKMeans with:

```text
n_clusters: 100
random_state: 42
```

### Reproduce Feature Engineering

The feature dataset can be regenerated using DVC:

```bash
dvc repro
```

The pipeline is defined in `dvc.yaml` and uses:

- `data/processed/NYC_clean.csv` as the input dataset
- `src/features/build_features.py` as the feature-engineering script
- `features.n_clusters` from `params.yaml`
- `data/features/nyc_features.csv` as the output

### Verify the Generated Dataset

The generated dataset can be inspected using Python:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/features/nyc_features.csv'); print(df.shape); print(df.columns.tolist())"
```

The large feature dataset is versioned using DVC rather than being stored directly in Git.

---

## Current Project Status

- Git repository configured
- GitHub repository connected
- DVC configured with Google Drive remote
- Raw dataset versioned
- Processed dataset versioned
- Feature-engineering pipeline implemented
- Feature dataset generated
- Feature dataset versioned with DVC
- DVC pipeline verified with `dvc repro`
- Collaborative project structure created

---

## Future Work

- Model training
- Hyperparameter tuning
- Model versioning
- Model evaluation
- CI/CD pipeline
- Model deployment

---

## License

Academic project for BITS Pilani Post Graduate Certificate (AI & ML).
