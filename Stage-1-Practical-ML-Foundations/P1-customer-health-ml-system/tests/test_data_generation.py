import pytest
from p1_customer_health.domain.synthetic_data import generate_customer_health_data
from p1_customer_health.domain.dataset import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE, CLASSIFICATION_TARGET, REGRESSION_TARGET


def test_shape_and_columns():
    df = generate_customer_health_data(n_samples=100, seed=1)
    assert len(df) == 100
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE, CLASSIFICATION_TARGET, REGRESSION_TARGET, "account_id", "snapshot_date"]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_churn_rate_is_realistic():
    df = generate_customer_health_data(n_samples=2000, seed=42)
    rate = df[CLASSIFICATION_TARGET].mean()
    # Expect churn rate between 5% and 30% — unrealistically high/low rates indicate broken generation
    assert 0.05 < rate < 0.30, f"Unexpected churn rate: {rate:.3f}"


def test_feature_bounds_are_valid():
    df = generate_customer_health_data(n_samples=500, seed=7)
    assert df["feature_adoption_ratio"].between(0.0, 1.0).all(), "feature_adoption_ratio out of [0, 1]"
    assert df["seat_utilization"].ge(0.0).all(), "seat_utilization has negative values"
    assert df["nps_score"].between(-100, 100).all(), "nps_score out of [-100, 100]"


def test_no_missing_values():
    df = generate_customer_health_data(n_samples=200, seed=3)
    assert df.isnull().sum().sum() == 0, "Unexpected missing values in generated data"
