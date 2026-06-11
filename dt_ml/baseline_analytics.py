# dt_ml/baseline_analytics.py
# dt_ml/baseline_analytics.py

import time
import logging
import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except Exception:
    seasonal_decompose = None


# =========================================================
# LOGGER
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# MAIN FUNCTION
# =========================================================

def compute(df: pd.DataFrame) -> dict:
    """
    Compute baseline business analytics.

    Features:
    - automatic schema detection
    - KPI aggregation
    - seasonality analysis
    - data quality scoring
    - UI-friendly warnings

    Performance Targets:
    - <2 sec for 10K rows
    - <10 sec for 100K rows

    Thread-safe:
    - no global mutable state

    Returns deterministic output.
    """

    # -----------------------------------------------------
    # TIMER
    # -----------------------------------------------------

    start_time = time.perf_counter()

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
    # CACHE
    # -----------------------------------------------------

    cache = {}

    # -----------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # -----------------------------------------------------

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # -----------------------------------------------------
    # WARNINGS
    # -----------------------------------------------------

    warnings = []

    if len(df) > 1000000:

        warnings.append(
            "Large dataset detected. Processing may be slower."
        )

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
    # EDGE CASE WARNINGS
    # -----------------------------------------------------

    if not revenue_col:

        warnings.append(
            "Revenue column missing."
        )

    if not date_col:

        warnings.append(
            "Date column missing."
        )

    missing_pct = df.isnull().mean()

    for col, pct in missing_pct.items():

        if pct > 0.15:

            warnings.append(
                f"{col.replace('_',' ').title()} has {pct:.0%} missing values."
            )

    # -----------------------------------------------------
    # DATE CLEANING
    # -----------------------------------------------------

    if date_col:

        if "parsed_dates" not in cache:

            cache["parsed_dates"] = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

        df[date_col] = cache["parsed_dates"]

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

            outlier_ratio = (
                detect_outliers(df[col]).mean()
            )

            if outlier_ratio > 0.05:

                warnings.append(
                    f"{col.replace('_',' ').title()} contains unusual values."
                )

    # -----------------------------------------------------
    # MONTHLY REVENUE AGGREGATION
    # -----------------------------------------------------

    monthly_revenue = []

    if revenue_col and date_col:

        if "monthly_df" not in cache:

            cache["monthly_df"] = (
                df.groupby(
                    pd.Grouper(
                        key=date_col,
                        freq="MS"
                    )
                )[revenue_col]
                .sum()
                .reset_index()
            )

        monthly_df = cache["monthly_df"]

        monthly_revenue = [
            {
                "month": month.strftime("%Y-%m"),
                "value": round(float(value), 2)
            }
            for month, value in zip(
                monthly_df[date_col],
                monthly_df[revenue_col]
            )
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
    # SORT + DEDUP WARNINGS
    # -----------------------------------------------------

    warnings = sorted(
        list(set(warnings))
    )

    # -----------------------------------------------------
    # RUNTIME
    # -----------------------------------------------------

    runtime = round(
        time.perf_counter() - start_time,
        2
    )

    logger.info(
        "compute completed | rows=%s | runtime=%.2fs",
        len(df),
        runtime
    )

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
        "runtime_seconds": runtime,
    }


# =========================================================
# HELPERS
# =========================================================

def detect_column(df, keywords):

    for col in df.columns:

        col_lower = str(col).lower()

        for key in keywords:

            if key in col_lower:
                return col

    return None


def clean_numeric(series):

    cleaned = (
        series.astype(str)
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    ).fillna(0)


def detect_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    return (
        (series < lower)
        | (series > upper)
    )


def calculate_quality_score(df):

    total_cells = df.size

    if total_cells == 0:
        return 0.0

    missing_ratio = (
        df.isna().sum().sum()
        / total_cells
    )

    completeness_score = (
        1 - missing_ratio
    )

    duplicate_score = (
        1 - df.duplicated().mean()
    )

    object_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    mixed_type_penalty = 0

    for col in object_cols:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        numeric_ratio = converted.notna().mean()

        if (
            numeric_ratio > 0.2
            and numeric_ratio < 0.8
        ):
            mixed_type_penalty += 0.05

    consistency_score = max(
        0,
        1 - mixed_type_penalty
    )

    score = (
        completeness_score * 0.45
        + duplicate_score * 0.35
        + consistency_score * 0.20
    )

    return round(
        min(max(score, 0), 1),
        2
    )


def empty_response():

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
        "runtime_seconds": 0.0,
    }
