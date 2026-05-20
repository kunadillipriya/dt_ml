import pandas as pd

from dt_ml.baseline_analytics import compute


def test_compute_empty_df():
    df = pd.DataFrame()

    result = compute(df)

    assert isinstance(result, dict)


def test_compute_returns_kpi_cards():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "revenue": [1000]
    })

    result = compute(df)

    assert "kpi_cards" in result


def test_growth_rate():
    df = pd.DataFrame({
        "date": [
            "2024-01-01",
            "2024-02-01"
        ],
        "revenue": [100, 200]
    })

    result = compute(df)

    assert result["growth_rate_pct"] == 100.0


def test_trend_growing():
    df = pd.DataFrame({
        "date": [
            "2024-01-01",
            "2024-02-01"
        ],
        "revenue": [100, 300]
    })

    result = compute(df)

    assert result["trend"] == "growing"


def test_churn_rate():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "revenue": [100],
        "churn_flag": [1]
    })

    result = compute(df)

    assert result["churn_rate_pct"] == 100.0


def test_salary_total():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "revenue": [100],
        "salary": [5000]
    })

    result = compute(df)

    assert result["headcount_cost_total"] == 5000.0


def test_marketing_cac():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "revenue": [100],
        "marketing_budget": [1000],
        "new_customers": [10]
    })

    result = compute(df)

    assert result["avg_marketing_cac"] == 100.0


def test_monthly_revenue_exists():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "revenue": [100]
    })

    result = compute(df)

    assert len(result["monthly_revenue"]) == 1