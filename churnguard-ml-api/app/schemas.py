from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ContractType = Literal["monthly", "quarterly", "annual"]
InternetService = Literal["dsl", "fiber", "wireless"]
PredictionLabel = Literal["churn", "stay"]


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, description="Customer tenure in months.")
    monthly_spend: float = Field(..., ge=0, le=500, description="Monthly subscription spend in USD.")
    support_tickets: int = Field(..., ge=0, le=50, description="Number of support tickets raised recently.")
    payment_delay_days: int = Field(..., ge=0, le=90, description="Average payment delay in days.")
    usage_hours_per_week: float = Field(..., ge=0, le=100, description="Estimated weekly product usage hours.")
    contract_type: ContractType = Field(..., description="Billing contract type.")
    internet_service: InternetService = Field(..., description="Primary internet service used by the customer.")
    is_premium: bool = Field(..., description="Whether the customer is on a premium plan.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 12,
                "monthly_spend": 89.0,
                "support_tickets": 3,
                "payment_delay_days": 7,
                "usage_hours_per_week": 8.0,
                "contract_type": "monthly",
                "internet_service": "fiber",
                "is_premium": False,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1, description="Predicted probability of churn.")
    prediction: PredictionLabel = Field(..., description="Binary prediction generated from the configured threshold.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    metrics_path: str
    threshold: float
    features: list[str]


class MetricsResponse(BaseModel):
    roc_auc: float
    accuracy: float
    train_rows: int
    test_rows: int
    threshold: float
    generated_at_utc: str
