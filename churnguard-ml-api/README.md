# ChurnGuard ML API

[![CI](https://github.com/Zakir060/churnguard-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Zakir060/churnguard-ml-api/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-black)

Production-style machine learning API that predicts **customer churn probability** from telecom-style behavioral features.

This project is designed as a portfolio-ready example of an **end-to-end ML product**, not just a notebook. It includes:

- synthetic but realistic tabular data generation
- reproducible training pipeline with scikit-learn
- persisted model artifacts and evaluation metrics
- FastAPI inference service with documented endpoints
- automated tests and GitHub Actions CI
- Docker support for containerized execution

> The dataset is synthetically generated so the full project runs locally without external downloads.

## Business goal

Customer churn is expensive. The goal of this project is to help a telecom-style business identify customers with a higher probability of leaving, so retention teams can intervene earlier with offers, support, or account actions.

### Inputs

- tenure in months
- monthly spend
- support ticket volume
- payment delay behavior
- weekly usage
- contract type
- internet service type
- premium plan status

### Outputs

- churn probability between 0 and 1
- binary prediction: `churn` or `stay`

## Architecture

```mermaid
flowchart LR
    A[Synthetic customer data] --> B[Training pipeline]
    B --> C[Saved model artifact]
    B --> D[Saved metrics]
    C --> E[FastAPI inference service]
    D --> E
    E --> F[/predict]
    E --> G[/model/metrics]
    E --> H[/health]
```

## Repository structure

```text
churnguard-ml-api/
├── app/
│   ├── api.py
│   ├── config.py
│   ├── data_generator.py
│   ├── predict.py
│   ├── schemas.py
│   └── train.py
├── models/
├── tests/
├── .github/workflows/ci.yml
├── .env.example
├── CONTRIBUTING.md
├── Dockerfile
├── example_request.json
├── Makefile
└── requirements.txt
```

## Model details

- **Model**: Logistic Regression
- **Preprocessing**:
  - median imputation + scaling for numeric features
  - most-frequent imputation + one-hot encoding for categorical features
- **Train/test split**: stratified 80/20
- **Default decision threshold**: `0.50`

The threshold can be changed with an environment variable:

```bash
MODEL_THRESHOLD=0.60
```

## Quickstart

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m app.train
```

This creates:

- `models/churn_model.joblib`
- `models/metrics.json`

### 4. Start the API

```bash
python -m uvicorn app.api:app --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/model/metrics`

## Example request

`example_request.json`

```json
{
  "tenure_months": 12,
  "monthly_spend": 89.0,
  "support_tickets": 3,
  "payment_delay_days": 7,
  "usage_hours_per_week": 8.0,
  "contract_type": "monthly",
  "internet_service": "fiber",
  "is_premium": false
}
```

cURL example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

Expected response shape:

```json
{
  "churn_probability": 0.18,
  "prediction": "stay"
}
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Basic service metadata |
| GET | `/health` | Liveness + model readiness |
| GET | `/model/info` | Model paths, threshold, features |
| GET | `/model/metrics` | Saved evaluation metrics |
| POST | `/predict` | Single-customer inference |

## Test suite

```bash
python -m pytest -q
```

## Docker

```bash
docker build -t churnguard-api .
docker run -p 8000:8000 churnguard-api
```

## CI

GitHub Actions automatically:

- installs dependencies
- trains the model
- runs the test suite on every push and pull request

## Roadmap

- add explainability with SHAP
- add batch inference endpoint
- log predictions and drift signals
- integrate a real churn dataset
- deploy on Render or Railway

## Why this project is strong for GitHub

This repository demonstrates that the author can:

- build a full ML pipeline, not only train a model in a notebook
- expose the model through a clean API layer
- structure a repository professionally
- persist artifacts and evaluation metrics
- test and automate the workflow with CI

## License

MIT
