from p1_customer_health.domain.synthetic_data import generate_customer_health_data


def test_generate_customer_health_data_has_expected_columns() -> None:
    df = generate_customer_health_data(n_samples=50, seed=1)
    assert len(df) == 50
    assert "churned_30d" in df.columns
    assert "revenue_change_next_30d" in df.columns
    assert "support_note" in df.columns
