# dt_ml/baseline_analytics.py
# dt_ml/baseline_analytics.py

import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except Exception:
    seasonal_decompose = None


# =========================================================
# MAIN FUNCTION
# =========================================================

def compute(df: pd.DataFrame) -> dict:
    """
    Returns baseline metrics for a dataset.

    Returns:
    {
        "monthly_revenue": [{"month":"2024-01","value":1240000}, ...],
        "growth_rate_pct": 8.4,
        "churn_rate_pct": 4.1,
        "avg_marketing_cac": 142.30,
        "headcount_cost_total": 285000,
        "trend": "growing",
        "seasonality_index": [...],
        "kpi_cards": {...},
        "data_quality_score": 0.0-1.0,
        "warnings": [...]
    }
    """

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if df is None:
        raise ValueError("Input dataframe cannot be None")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # -----------------------------------------------------
    # EMPTY DATAFRAME
    # -----------------------------------------------------

    if df.empty:
        return empty_response()

    # -----------------------------------------------------
    # COPY DATAFRAME
    # -----------------------------------------------------

    df = df.copy()

    # -----------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # -----------------------------------------------------

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    warnings = []

    # -----------------------------------------------------
    # AUTO DETECT COLUMNS
    # -----------------------------------------------------

    date_col = detect_column(
        df,
        ["date", "month", "time", "period"]
    )

    revenue_col = detect_column(
        df,
        ["revenue", "sales", "income"]
    )

    churn_col = detect_column(
        df,
        ["churn", "churn_flag", "cancelled", "lost"]
    )

    marketing_col = detect_column(
        df,
        ["marketing", "budget", "ad_spend", "campaign"]
    )

    salary_col = detect_column(
        df,
        ["salary", "wage", "comp"]
    )

    new_customer_col = detect_column(
        df,
        ["new_customers", "conversions", "leads"]
    )

    customer_col = detect_column(
        df,
        ["customer_id", "customer", "client"]
    )

    # -----------------------------------------------------
    # DATE CLEANING
    # -----------------------------------------------------

    if date_col:

        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        invalid_dates = df[date_col].isna().sum()

        if invalid_dates > 0:
            warnings.append(
                f"{invalid_dates} invalid date rows removed."
            )

        df = df.dropna(subset=[date_col])

    else:
        warnings.append(
            "No date column detected."
        )

    # -----------------------------------------------------
    # NUMERIC CLEANING
    # -----------------------------------------------------

    numeric_candidates = [
        revenue_col,
        churn_col,
        marketing_col,
        salary_col,
        new_customer_col
    ]

    for col in numeric_candidates:

        if col and col in df.columns:
            df[col] = clean_numeric(df[col])

    # -----------------------------------------------------
    # MONTHLY REVENUE AGGREGATION
    # -----------------------------------------------------

    monthly_revenue = []

    if revenue_col and date_col:

        monthly_df = (
            df.groupby(
                pd.Grouper(
                    key=date_col,
                    freq="MS"
                )
            )[revenue_col]
            .sum()
            .reset_index()
        )

        monthly_revenue = [
            {
                "month": row[date_col].strftime("%Y-%m"),
                "value": round(float(row[revenue_col]), 2)
            }
            for _, row in monthly_df.iterrows()
        ]

    else:
        warnings.append(
            "Monthly revenue could not be computed."
        )

    # -----------------------------------------------------
    # GROWTH RATE
    # -----------------------------------------------------

    growth_rate_pct = 0.0

    if len(monthly_revenue) >= 2:

        first = monthly_revenue[0]["value"]
        last = monthly_revenue[-1]["value"]

        if first != 0:

            growth_rate_pct = round(
                ((last - first) / first) * 100,
                2
            )

    # -----------------------------------------------------
    # CHURN RATE
    # -----------------------------------------------------

    churn_rate_pct = None

    if churn_col:

        churn_rate_pct = round(
            float(df[churn_col].mean() * 100),
            2
        )

    elif customer_col:

        unique_customers = df[customer_col].nunique()

        if unique_customers > 0:

            churn_rate_pct = round(
                (1 / unique_customers) * 100,
                2
            )

            warnings.append(
                "Estimated churn rate from customer reuse pattern."
            )

    else:
        warnings.append(
            "Churn rate unavailable."
        )

    # -----------------------------------------------------
    # MARKETING CAC
    # -----------------------------------------------------

    avg_marketing_cac = None

    if marketing_col and new_customer_col:

        total_spend = df[marketing_col].sum()

        total_new_customers = df[new_customer_col].sum()

        if total_new_customers > 0:

            avg_marketing_cac = round(
                float(total_spend / total_new_customers),
                2
            )

    else:
        warnings.append(
            "Marketing CAC unavailable."
        )

    # -----------------------------------------------------
    # HEADCOUNT COST
    # -----------------------------------------------------

    headcount_cost_total = None

    if salary_col:

        headcount_cost_total = round(
            float(df[salary_col].sum()),
            2
        )

    else:
        warnings.append(
            "Salary data unavailable."
        )

    # -----------------------------------------------------
    # TREND DETECTION
    # -----------------------------------------------------

    if growth_rate_pct > 5:
        trend = "growing"

    elif growth_rate_pct < -5:
        trend = "declining"

    else:
        trend = "flat"

    # -----------------------------------------------------
    # SEASONALITY INDEX
    # -----------------------------------------------------

    seasonality_index = None

    if (
        seasonal_decompose is not None
        and len(monthly_revenue) >= 24
    ):

        try:

            ts = pd.Series([
                x["value"]
                for x in monthly_revenue
            ])

            decomposition = seasonal_decompose(
                ts,
                model="multiplicative",
                period=12
            )

            seasonality_index = [
                round(float(x), 2)
                for x in decomposition.seasonal[:12]
            ]

        except Exception:

            warnings.append(
                "Seasonality calculation failed."
            )

    # -----------------------------------------------------
    # DATA QUALITY SCORE
    # -----------------------------------------------------

    data_quality_score = calculate_quality_score(df)

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    latest_monthly_revenue = (
        monthly_revenue[-1]["value"]
        if monthly_revenue
        else 0
    )

    kpi_cards = {
        "monthly_revenue": latest_monthly_revenue,
        "growth_rate_pct": growth_rate_pct,
        "churn_rate_pct": churn_rate_pct,
        "avg_marketing_cac": avg_marketing_cac,
        "headcount_cost_total": headcount_cost_total,
        "trend": trend,
        "data_quality_score": data_quality_score,
    }

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    return {
        "monthly_revenue": monthly_revenue,
        "growth_rate_pct": growth_rate_pct,
        "churn_rate_pct": churn_rate_pct,
        "avg_marketing_cac": avg_marketing_cac,
        "headcount_cost_total": headcount_cost_total,
        "trend": trend,
        "seasonality_index": seasonality_index,
        "kpi_cards": kpi_cards,
        "data_quality_score": data_quality_score,
        "warnings": warnings,
    }


# =========================================================
# HELPERS
# =========================================================

def detect_column(df, keywords):
    """
    Detect columns using keyword matching.
    """

    for col in df.columns:

        col_lower = str(col).lower()

        for key in keywords:

            if key in col_lower:
                return col

    return None


def clean_numeric(series):
    """
    Convert messy numeric strings into float.
    """

    return (
        series.astype(str)
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
        .fillna(0)
    )


def calculate_quality_score(df):
    """
    Composite quality score:
    completeness + duplicates + consistency
    """

    total_cells = df.size

    if total_cells == 0:
        return 0.0

    # ---------------------------------------------
    # Completeness
    # ---------------------------------------------

    missing_cells = df.isna().sum().sum()

    completeness = 1 - (
        missing_cells / total_cells
    )

    # ---------------------------------------------
    # Duplicate score
    # ---------------------------------------------

    duplicate_score = 1 - (
        df.duplicated().mean()
    )

    # ---------------------------------------------
    # Consistency score
    # ---------------------------------------------

    consistency_score = 1.0

    for col in df.columns:

        if df[col].dtype == "object":

            try:
                pd.to_numeric(df[col])

            except Exception:
                consistency_score *= 0.98

    # ---------------------------------------------
    # Final weighted score
    # ---------------------------------------------

    score = (
        (0.5 * completeness)
        + (0.3 * duplicate_score)
        + (0.2 * consistency_score)
    )

    return round(float(score), 2)


def empty_response():
    """
    Standard empty response structure.
    """

    return {
        "monthly_revenue": [],
        "growth_rate_pct": 0.0,
        "churn_rate_pct": None,
        "avg_marketing_cac": None,
        "headcount_cost_total": None,
        "trend": "flat",
        "seasonality_index": None,
        "kpi_cards": {},
        "data_quality_score": 0.0,
        "warnings": [
            "Empty dataframe provided."
        ],
    }