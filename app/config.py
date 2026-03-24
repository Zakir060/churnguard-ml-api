from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
RANDOM_STATE = 42
