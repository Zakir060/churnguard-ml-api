from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import METRICS_PATH, MODEL_PATH, MODEL_THRESHOLD


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            "Model file not found. Run `python -m app.train` before starting the API."
        )
    return joblib.load(MODEL_PATH)


def model_ready() -> bool:
    return MODEL_PATH.exists()


def read_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise ModelNotFoundError(
            "Metrics file not found. Run `python -m app.train` before requesting model metrics."
        )
    return json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))


def predict_single(payload: dict) -> tuple[float, str]:
    model = load_model()
    df = pd.DataFrame([payload])
    probability = float(model.predict_proba(df)[0, 1])
    label = "churn" if probability >= MODEL_THRESHOLD else "stay"
    return round(probability, 4), label
