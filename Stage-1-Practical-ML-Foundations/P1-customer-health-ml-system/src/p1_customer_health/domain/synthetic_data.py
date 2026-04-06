from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _make_support_note(rng: np.random.Generator, risk_level: float, plan_type: str, tickets_30d: int) -> str:
    healthy_phrases = [
        "team using dashboard daily",
        "positive onboarding feedback",
        "expansion discussion started",
        "feature adoption improving",
    ]
    risk_phrases = [
        "slow response complaint",
        "billing frustration reported",
        "missing integration blocker",
        "low usage and unclear value",
    ]
    neutral = f"{plan_type} account with {tickets_30d} support tickets this month"
    phrase = rng.choice(risk_phrases if risk_level > 0.55 else healthy_phrases)
    return f"{neutral}; {phrase}"


def generate_customer_health_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = datetime(2025, 1, 1)

    plan_type = rng.choice(["starter", "growth", "enterprise"], size=n_samples, p=[0.45, 0.4, 0.15])
    industry = rng.choice(["saas", "retail", "finance", "healthcare", "education"], size=n_samples)
    region = rng.choice(["NA", "EU", "GCC", "APAC"], size=n_samples)
    contract_type = rng.choice(["monthly", "annual"], size=n_samples, p=[0.55, 0.45])

    account_age_days = rng.integers(30, 1800, size=n_samples)
    days_since_last_activity = np.clip(rng.normal(12, 10, size=n_samples), 0, 90).astype(int)
    logins_30d = np.clip(rng.normal(18, 9, size=n_samples), 0, 60).astype(int)
    tickets_30d = np.clip(rng.poisson(2.4, size=n_samples), 0, 15)
    feature_adoption_ratio = np.clip(rng.normal(0.58, 0.18, size=n_samples), 0.02, 0.99)
    nps_score = np.clip(rng.normal(24, 28, size=n_samples), -100, 100).round(0).astype(int)
    payment_failures_90d = np.clip(rng.poisson(0.4, size=n_samples), 0, 5)
    seat_utilization = np.clip(rng.normal(0.68, 0.2, size=n_samples), 0.05, 1.2)

    spend_base = {
        "starter": 180,
        "growth": 650,
        "enterprise": 2200,
    }
    monthly_spend = np.array([spend_base[p] for p in plan_type], dtype=float)
    monthly_spend += rng.normal(0, 65, size=n_samples)
    monthly_spend += (seat_utilization - 0.65) * 230
    monthly_spend = np.clip(monthly_spend, 50, None).round(2)

    plan_risk = np.select(
        [plan_type == "starter", plan_type == "growth", plan_type == "enterprise"],
        [0.18, 0.1, 0.06],
    )
    contract_risk = np.where(contract_type == "monthly", 0.1, -0.05)
    activity_risk = days_since_last_activity / 140
    ticket_risk = tickets_30d / 26
    payment_risk = payment_failures_90d / 10
    adoption_relief = feature_adoption_ratio * 0.35
    nps_relief = np.clip((nps_score + 100) / 200, 0, 1) * 0.2
    utilization_relief = seat_utilization * 0.12

    raw_risk = plan_risk + contract_risk + activity_risk + ticket_risk + payment_risk - adoption_relief - nps_relief - utilization_relief
    churn_probability = 1 / (1 + np.exp(-(raw_risk * 4 - 1.6)))
    churn_probability = np.clip(churn_probability, 0.03, 0.82)
    churned_30d = rng.binomial(1, churn_probability, size=n_samples)

    revenue_noise = rng.normal(0, 110, size=n_samples)
    expansion_signal = (feature_adoption_ratio - 0.45) * 320 + (seat_utilization - 0.6) * 280
    contraction_signal = -days_since_last_activity * 1.6 - tickets_30d * 9 - payment_failures_90d * 18
    churn_penalty = churned_30d * rng.uniform(250, 850, size=n_samples)
    revenue_change_next_30d = (expansion_signal + contraction_signal - churn_penalty + revenue_noise).round(2)

    snapshot_offsets = rng.integers(0, 240, size=n_samples)
    snapshot_date = [base_date + timedelta(days=int(offset)) for offset in snapshot_offsets]
    support_note = [
        _make_support_note(rng, float(prob), str(plan), int(tickets))
        for prob, plan, tickets in zip(churn_probability, plan_type, tickets_30d)
    ]

    return pd.DataFrame(
        {
            "account_id": [f"acct_{i:05d}" for i in range(n_samples)],
            "snapshot_date": snapshot_date,
            "plan_type": plan_type,
            "industry": industry,
            "region": region,
            "contract_type": contract_type,
            "monthly_spend": monthly_spend,
            "logins_30d": logins_30d,
            "tickets_30d": tickets_30d,
            "feature_adoption_ratio": feature_adoption_ratio.round(3),
            "days_since_last_activity": days_since_last_activity,
            "account_age_days": account_age_days,
            "nps_score": nps_score,
            "payment_failures_90d": payment_failures_90d,
            "seat_utilization": seat_utilization.round(3),
            "support_note": support_note,
            "churned_30d": churned_30d,
            "revenue_change_next_30d": revenue_change_next_30d,
        }
    )
