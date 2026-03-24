from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import APP_DESCRIPTION, APP_NAME, APP_VERSION, METRICS_PATH, MODEL_PATH, MODEL_THRESHOLD
from app.predict import ModelNotFoundError, model_ready, predict_single, read_metrics
from app.schemas import (
    CustomerFeatures,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from app.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    contact={"name": "Zakir Bayramov", "url": "https://github.com/Zakir060"},
    license_info={"name": "MIT"},
)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_ready=model_ready())


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def get_model_info() -> ModelInfoResponse:
    features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    return ModelInfoResponse(
        model_path=str(MODEL_PATH),
        metrics_path=str(METRICS_PATH),
        threshold=MODEL_THRESHOLD,
        features=features,
    )


@app.get("/model/metrics", response_model=MetricsResponse, tags=["meta"])
def get_model_metrics() -> MetricsResponse:
    try:
        metrics = read_metrics()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MetricsResponse(**metrics)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: CustomerFeatures) -> PredictionResponse:
    try:
        probability, label = predict_single(payload.model_dump())
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(churn_probability=probability, prediction=label)
