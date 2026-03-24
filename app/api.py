from fastapi import FastAPI, HTTPException

from app.predict import ModelNotFoundError, predict_single
from app.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="ChurnGuard ML API",
    version="1.0.0",
    description="Predict telecom customer churn probability using a trained scikit-learn model.",
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    try:
        probability, label = predict_single(payload.model_dump())
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(churn_probability=probability, prediction=label)
