# ChurnGuard ML API

A portfolio-ready machine learning project that predicts **customer churn probability** from telecom-style customer behavior features.

This repository includes:
- synthetic but realistic tabular data generation
- feature engineering + training pipeline with scikit-learn
- model evaluation and saved artifacts
- FastAPI inference service
- unit tests
- Docker support
- GitHub Actions CI

> Note: The dataset in this project is **synthetically generated** for demonstration purposes, so the full pipeline works out of the box without external downloads.

## Project structure

```bash
churnguard-ml-api/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── data_generator.py
│   ├── predict.py
│   ├── schemas.py
│   └── train.py
├── models/
│   └── .gitkeep
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── .gitignore
├── Dockerfile
├── LICENSE
├── Makefile
├── example_request.json
└── requirements.txt
```

## Features used

- `tenure_months`
- `monthly_spend`
- `support_tickets`
- `payment_delay_days`
- `usage_hours_per_week`
- `contract_type`
- `internet_service`
- `is_premium`

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m app.train
```

This will generate:
- `models/churn_model.joblib`
- `models/metrics.json`

### 4. Run the API

```bash
uvicorn app.api:app --reload
```

Open:
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

## Example API request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

Expected response:

```json
{
  "churn_probability": 0.18,
  "prediction": "stay"
}
```

## Run tests

```bash
pytest
```

## Docker

```bash
docker build -t churnguard-api .
docker run -p 8000:8000 churnguard-api
```

## Why this project is good for GitHub

- shows end-to-end ML workflow, not just a notebook
- includes a production-style API layer
- has tests and CI
- works offline thanks to synthetic data generation
- easy to extend with real datasets later

## Ideas to improve it later

- add model explainability with SHAP
- store experiments with MLflow
- add Streamlit dashboard
- replace synthetic data with a real telecom churn dataset
- deploy on Render, Railway, or Hugging Face Spaces

## License

MIT
