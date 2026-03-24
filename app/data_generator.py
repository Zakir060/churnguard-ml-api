from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import RANDOM_STATE


CONTRACT_TYPES = ["monthly", "quarterly", "annual"]
INTERNET_SERVICES = ["dsl", "fiber", "wireless"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customer_data(n_samples: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic telecom-style customer churn dataset."""
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=n_samples)
    monthly_spend = np.clip(rng.normal(75, 25, size=n_samples), 15, 180)
    support_tickets = rng.poisson(1.8, size=n_samples)
    payment_delay_days = np.clip(rng.normal(4, 6, size=n_samples), 0, 35).round().astype(int)
    usage_hours_per_week = np.clip(rng.normal(14, 7, size=n_samples), 1, 45)
    contract_type = rng.choice(CONTRACT_TYPES, size=n_samples, p=[0.58, 0.22, 0.20])
    internet_service = rng.choice(INTERNET_SERVICES, size=n_samples, p=[0.32, 0.48, 0.20])
    is_premium = rng.choice([0, 1], size=n_samples, p=[0.68, 0.32])

    logit = (
        -1.5
        + 0.02 * monthly_spend
        + 0.22 * support_tickets
        + 0.09 * payment_delay_days
        - 0.03 * tenure_months
        - 0.04 * usage_hours_per_week
        - 0.45 * is_premium
        + np.where(contract_type == "monthly", 0.85, 0)
        + np.where(contract_type == "annual", -0.65, 0)
        + np.where(internet_service == "fiber", 0.22, 0)
        + rng.normal(0, 0.55, size=n_samples)
    )

    churn_probability = _sigmoid(logit)
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets": support_tickets,
            "payment_delay_days": payment_delay_days,
            "usage_hours_per_week": usage_hours_per_week.round(2),
            "contract_type": contract_type,
            "internet_service": internet_service,
            "is_premium": is_premium.astype(bool),
            "churn": churn,
        }
    )
