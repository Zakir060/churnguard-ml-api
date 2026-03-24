import json

from app.config import METRICS_PATH, MODEL_PATH
from app.predict import predict_single
from app.train import train_and_save


def test_training_creates_artifacts():
    metrics = train_and_save()
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_prediction_output_schema():
    train_and_save()
    probability, label = predict_single(
        {
            "tenure_months": 12,
            "monthly_spend": 89.0,
            "support_tickets": 3,
            "payment_delay_days": 7,
            "usage_hours_per_week": 8.0,
            "contract_type": "monthly",
            "internet_service": "fiber",
            "is_premium": False,
        }
    )
    assert isinstance(probability, float)
    assert label in {"churn", "stay"}


def test_metrics_file_contains_expected_keys():
    train_and_save()
    data = json.loads(METRICS_PATH.read_text())
    assert {"roc_auc", "accuracy", "train_rows", "test_rows"}.issubset(data.keys())
