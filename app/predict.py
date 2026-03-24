from __future__ import annotations

import pandas as pd
import joblib

from app.config import MODEL_PATH


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model():
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            "Model file not found. Run `python -m app.train` before starting the API."
        )
    return joblib.load(MODEL_PATH)


def predict_single(payload: dict) -> tuple[float, str]:
    model = load_model()
    df = pd.DataFrame([payload])
    probability = float(model.predict_proba(df)[0, 1])
    label = "churn" if probability >= 0.5 else "stay"
    return round(probability, 4), label
