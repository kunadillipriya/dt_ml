# dt_ml/data_loader.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np


# ---------------------------------------------------
# COLUMN ROLE PATTERNS
# ---------------------------------------------------

COLUMN_PATTERNS = {
    "date": r"date|time|month|period",
    "revenue": r"revenue|sales|income|gross",
    "price": r"price|unit_price|cost_per_unit|rate",
    "units_sold": r"qty|quantity|units|volume",
    "customer_id": r"customer|client|user|account",
    "churn": r"churn|cancelled|lost",
    "marketing_spend": r"marketing|ad_spend|campaign",
    "salary": r"salary|wage|compensation",
    "headcount": r"employee|headcount|staff",
}


# ---------------------------------------------------
# LOAD CSV
# ---------------------------------------------------

def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Robust CSV loader for varying quality datasets.
    """

    encodings = ["utf-8", "latin1", "cp1252"]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
            )

            return normalize_dataframe(df)

        except Exception as e:
            last_error = e

    raise ValueError(f"Unable to load CSV: {last_error}")


# ---------------------------------------------------
# NORMALIZE DATAFRAME
# ---------------------------------------------------

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # remove duplicates
    df = df.drop_duplicates()

    return df


# ---------------------------------------------------
# COLUMN ROLE DETECTION
# ---------------------------------------------------

def detect_column_roles(
    df: pd.DataFrame
) -> Dict[str, str]:

    roles = {}

    for col in df.columns:

        col_name = col.lower()

        for role, pattern in COLUMN_PATTERNS.items():

            if re.search(pattern, col_name):

                # DATE CHECK
                if role == "date":
                    try:
                        pd.to_datetime(df[col])
                        roles[col] = role
                        break
                    except Exception:
                        continue

                # NUMERIC CHECK
                elif role in [
                    "revenue",
                    "price",
                    "units_sold",
                    "marketing_spend",
                    "salary",
                    "headcount",
                ]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        roles[col] = role
                        break

                # CUSTOMER ID CHECK
                elif role == "customer_id":
                    if df[col].nunique() > len(df) * 0.5:
                        roles[col] = role
                        break

                # CHURN CHECK
                elif role == "churn":

                    unique_vals = set(
                        df[col]
                        .dropna()
                        .astype(str)
                        .unique()
                    )

                    if unique_vals.issubset(
                        {"0", "1", "True", "False"}
                    ):
                        roles[col] = role
                        break

                else:
                    roles[col] = role
                    break

    return roles


# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------

def summarize_dataframe(
    df: pd.DataFrame
) -> Dict[str, Any]:

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }