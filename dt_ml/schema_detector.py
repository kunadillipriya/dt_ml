# dt_ml/schema_detector.py

import re
import pandas as pd


ROLE_PATTERNS = {
    "date": r"date|time|month|period",
    "revenue": r"revenue|sales|income|gross",
    "price": r"price|cost_per_unit|rate",
    "units_sold": r"qty|quantity|units|volume",
    "customer_id": r"customer|client|user|account",
    "churn": r"churn|cancelled|lost",
    "marketing_spend": r"marketing|ad_spend|campaign|budget",
    "salary": r"salary|wage|comp",
}


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Detect business roles for dataframe columns.
    """

    detected = {}

    for col in df.columns:

        for role, pattern in ROLE_PATTERNS.items():

            if re.search(pattern, col):

                detected[role] = col

    return detected