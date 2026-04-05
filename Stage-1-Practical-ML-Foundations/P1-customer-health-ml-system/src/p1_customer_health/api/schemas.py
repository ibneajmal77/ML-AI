from datetime import datetime

from pydantic import BaseModel, Field


class CustomerHealthRecord(BaseModel):
    account_id: str
    snapshot_date: datetime
    plan_type: str
    industry: str
    region: str
    contract_type: str
    monthly_spend: float
    logins_30d: int
    tickets_30d: int
    feature_adoption_ratio: float = Field(ge=0.0, le=1.5)
    days_since_last_activity: int
    account_age_days: int
    nps_score: int
    payment_failures_90d: int
    seat_utilization: float = Field(ge=0.0, le=1.5)
    support_note: str


class PredictRequest(BaseModel):
    records: list[CustomerHealthRecord]


class PredictionRow(BaseModel):
    account_id: str
    churn_score: float
    churn_label: int
    predicted_revenue_change: float
    segment_id: int
    anomaly_flag: int


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[PredictionRow]
