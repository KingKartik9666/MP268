# MP268 – Fuel Station Daily Sales Prediction (Surat)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLOps](https://img.shields.io/badge/MLOps-SCSE3040-orange)
![Group](https://img.shields.io/badge/Group-G252-purple)
![Problem](https://img.shields.io/badge/Problem-MP268-green)

## Project Details

| Field | Details |
|---|---|
| Group ID | G252 |
| Problem ID | MP268 |
| Problem Title | Fuel Station Daily Sales Prediction (Surat) |
| Course | SCSE3040 MLOps |
| Batch | 76 |
| Faculty | Dr. Upasna Joshi |

## Team Members

| Enrollment Number | Name |
|---|---|
| S24CSEU1990 | Anuj Gupta |
| S24CSEU0077 | Tarun Kirade |
| S24CSEU1982 | Kartik Negi |
| S24CSEU0941 | Shubh Pratap Parihar Singh |

---

## Project Overview

Fuel stations require reliable daily sales estimates to manage fuel inventory, operational resources and planning. Inaccurate sales forecasts can lead to inefficient inventory management, resource allocation issues and operational challenges.

This project focuses on developing a machine learning-based solution for predicting fuel station daily sales in Surat using the provided dataset. The project follows an MLOps-oriented workflow involving data quality analysis, preprocessing, model development, testing, deployment and monitoring.

The aim is to develop a reproducible and maintainable machine learning workflow that integrates data validation, model experimentation, automated testing and deployment-oriented infrastructure.

## Objectives

- Analyze the provided fuel station daily sales dataset.
- Identify data quality issues and inconsistencies.
- Develop data cleaning and preprocessing pipelines.
- Build and evaluate a machine learning model for sales prediction.
- Implement a prediction API using FastAPI.
- Validate the system using automated tests.
- Automate testing through GitHub Actions.
- Containerize the application using Docker.
- Explore Kubernetes deployment.
- Track datasets and experiments using MLOps tools.
- Monitor model performance and potential data drift.

## Project Workflow

The project follows an MLOps workflow:

1. Dataset inspection and exploration
2. Data quality checks
3. Data cleaning and preprocessing
4. Feature preparation
5. Model training and experimentation
6. Model evaluation
7. Model registration and versioning
8. API development
9. Automated testing and CI
10. Docker containerization
11. Kubernetes deployment
12. Monitoring and retraining

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data analysis and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Pytest | Unit and API testing |
| FastAPI | API development |
| Pydantic | Input validation |
| Git | Version control |
| GitHub | Collaborative development |
| GitHub Actions | Continuous integration |
| Docker | Application containerization |
| Kubernetes | Container orchestration |
| DVC | Dataset versioning |
| MLflow | Experiment tracking and model lifecycle management |
| YAML | Configuration management |

## Repository Structure

```text
MP268/
│
├── app/
│   └── main.py
│
├── src/
│   └── mp268/
│       ├── __init__.py
│       └── data_checks.py
│
├── tests/
│
├── data/
│
├── models/
│
├── output/
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── params.yaml
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

*The repository structure may change as the project implementation progresses.*

## Dataset

The project dataset is provided as part of the MP268 assignment.

The dataset files include:

- `train.csv` – Training dataset
- `later.csv` – Later-period data for subsequent analysis
- `holdout_features.csv` – Holdout data for prediction and evaluation

The dataset is used for data quality analysis, preprocessing, model development and performance evaluation.

> The original dataset files are handled according to the assignment guidelines. The original training data should not be modified manually in Excel.

## Data Quality Analysis

Data quality analysis is an important step before training the machine learning model.

The project examines potential data issues, including:

- Duplicate records
- Spelling inconsistencies
- Unexpected category values
- Missing values
- Sentinel or invalid values
- Data leakage
- Target variable inconsistencies
- Unit inconsistencies
- Sensor anomalies

Data checks are used to identify issues and support appropriate preprocessing and cleaning decisions.

## Data Cleaning and Preprocessing

The preprocessing pipeline aims to improve dataset quality while preserving the integrity of the original data.

Potential preprocessing operations include:

- Handling duplicate records
- Standardizing inconsistent categories
- Handling missing values
- Validating numerical features
- Detecting anomalous values
- Checking feature consistency
- Preparing data for model training

Cleaning decisions are documented and applied through code rather than manually altering the original dataset.

## Machine Learning

The project aims to develop a supervised machine learning model for fuel station daily sales prediction.

The machine learning workflow includes:

- Data preparation
- Feature engineering
- Training and validation
- Model experimentation
- Model evaluation
- Prediction generation
- Model versioning

### Evaluation Metrics

Regression metrics may include:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The final evaluation results will be documented after the experiments are completed.

## API Development

FastAPI is used to build a prediction service.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health status of the API |
| `/predict` | POST | Generate sales predictions |

The API uses input validation to ensure that incoming requests meet the expected schema and constraints.

API functionality is tested using automated tests.

## Testing

Automated testing is implemented using Pytest.

Testing focuses on:

- Data quality check functions
- Input validation
- API functionality
- Prediction endpoint behavior
- Health endpoint behavior
- Error handling
- Independent function testing

Tests are designed to support reliability and help detect problems during development and integration.

## Continuous Integration

GitHub Actions is used for continuous integration.

The CI workflow is designed to:

1. Trigger on repository pushes.
2. Set up the Python environment.
3. Install project dependencies.
4. Run automated tests.
5. Report test results.

Continuous integration helps detect errors early and maintain code quality across contributions.

## Docker

Docker is used to containerize the application.

The containerization process includes:

- Using a lightweight Python base image.
- Installing required dependencies.
- Copying the application files.
- Exposing the application port.
- Running the API using Uvicorn.

This provides a consistent environment for running the application.

## Kubernetes

Kubernetes deployment configuration is used to explore container orchestration.

The deployment configuration is intended to include:

- Deployment replicas
- Container image configuration
- Readiness probes
- Liveness probes

A Kubernetes Service is used to expose the API and connect requests to the application pods.

## MLOps

The project incorporates MLOps concepts to support reproducibility, monitoring and model lifecycle management.

The MLOps workflow explores:

- Dataset versioning using DVC
- Experiment tracking using MLflow
- Model registration
- Model versioning
- Model performance monitoring
- Data drift detection
- Model retraining
- Model promotion decisions

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/KingKartik9666/MP268.git
cd MP268
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the tests

```bash
pytest
```

### 6. Run the API

Once the application is configured:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Access the API

- Health endpoint: `http://localhost:8000/health`
- Interactive API documentation: `http://localhost:8000/docs`

## GitHub Repository

[https://github.com/KingKartik9666/MP268](https://github.com/KingKartik9666/MP268)

## Academic Submission

This project is developed as part of the SCSE3040 MLOps mini project assignment for Batch 76.

Every group member contributes through individual commits, collaborative development, testing and project implementation.

---

**G252 | MP268 | SCSE3040 MLOps | Batch 76**

Fuel Station Daily Sales Prediction – Surat
