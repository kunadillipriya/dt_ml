# tests/test_data_loader.py

import pandas as pd

from dt_ml.data_loader import (
    load_csv,
    detect_column_roles,
)

from dt_ml.data_quality import (
    compute_data_quality,
)


# ---------------------------------------------------
# SALES DATASET
# ---------------------------------------------------

def test_sales_dataset():

    df = load_csv("validation/test_data/sales_data.csv")

    roles = detect_column_roles(df)

    assert isinstance(roles, dict)

    quality = compute_data_quality(df)

    assert 0 <= quality["score"] <= 1


# ---------------------------------------------------
# OPS DATASET
# ---------------------------------------------------

def test_ops_dataset():

    df = load_csv("validation/test_data/ops_data.csv")

    roles = detect_column_roles(df)

    assert isinstance(roles, dict)

    quality = compute_data_quality(df)

    assert 0 <= quality["score"] <= 1


# ---------------------------------------------------
# MARKETING DATASET
# ---------------------------------------------------

def test_marketing_dataset():

    df = load_csv("validation/test_data/marketing_data.csv")

    roles = detect_column_roles(df)

    assert isinstance(roles, dict)

    quality = compute_data_quality(df)

    assert 0 <= quality["score"] <= 1


# ---------------------------------------------------
# CUSTOMER DATASET
# ---------------------------------------------------

def test_customer_dataset():

    df = load_csv("validation/test_data/customer_data.csv")

    roles = detect_column_roles(df)

    assert isinstance(roles, dict)

    quality = compute_data_quality(df)

    assert 0 <= quality["score"] <= 1


# ---------------------------------------------------
# FINANCIAL DATASET
# ---------------------------------------------------

def test_financial_dataset():

    df = load_csv("validation/test_data/financials.csv")

    roles = detect_column_roles(df)

    assert isinstance(roles, dict)

    quality = compute_data_quality(df)

    assert 0 <= quality["score"] <= 1


# ---------------------------------------------------
# HR DATASET
# ---------------------------------------------------

def test_hr_dataset():

    df = load_csv("validation/test_data/hr_data.csv")

    roles = detect_column_roles(df)

    assert isinstance(roles, dict)

    quality = compute_data_quality(df)

    assert 0 <= quality["score"] <= 1


# ---------------------------------------------------
# ROLE DETECTION
# ---------------------------------------------------

def test_detect_roles():

    df = pd.DataFrame(
        {
            "sale_date": pd.date_range(
                "2025-01-01",
                periods=5,
            ),
            "revenue": [100, 200, 300, 400, 500],
            "customer_id": [1, 2, 3, 4, 5],
        }
    )

    roles = detect_column_roles(df)

    assert roles["sale_date"] == "date"
    assert roles["revenue"] == "revenue"
    assert roles["customer_id"] == "customer_id"


# ---------------------------------------------------
# QUALITY WARNINGS
# ---------------------------------------------------

def test_quality_warnings():

    df = pd.DataFrame(
        {
            "revenue": [100, 200, None, 50000],
            "sales": [1, 2, 3, 4],
        }
    )

    quality = compute_data_quality(df)

    assert "warnings" in quality