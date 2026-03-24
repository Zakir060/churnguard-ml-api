from __future__ import annotations

import os
from pathlib import Path

from app import __version__

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "churn_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", MODELS_DIR / "metrics.json"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.50"))
APP_NAME = "ChurnGuard ML API"
APP_VERSION = __version__
APP_DESCRIPTION = (
    "Predict telecom-style customer churn probability using a trained scikit-learn pipeline. "
    "This project demonstrates an end-to-end ML workflow: synthetic data generation, model "
    "training, artifact persistence, automated tests, and a FastAPI inference service."
)
