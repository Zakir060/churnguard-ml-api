from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_spend: float = Field(..., ge=0, le=500)
    support_tickets: int = Field(..., ge=0, le=50)
    payment_delay_days: int = Field(..., ge=0, le=90)
    usage_hours_per_week: float = Field(..., ge=0, le=100)
    contract_type: str = Field(..., examples=["monthly", "quarterly", "annual"])
    internet_service: str = Field(..., examples=["dsl", "fiber", "wireless"])
    is_premium: bool


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
