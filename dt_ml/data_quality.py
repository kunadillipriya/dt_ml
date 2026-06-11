# dt_ml/data_quality.py
from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd
import numpy as np


def compute_data_quality(
    df: pd.DataFrame
) -> Dict[str, Any]:

    warnings: List[str] = []

    # ---------------------------------------------------
    # COMPLETENESS SCORE
    # ---------------------------------------------------

    total_cells = df.shape[0] * df.shape[1]

    missing_cells = df.isna().sum().sum()

    completeness_score = (
        1 - (missing_cells / total_cells)
        if total_cells > 0
        else 0
    )

    # ---------------------------------------------------
    # DUPLICATE SCORE
    # ---------------------------------------------------

    duplicate_ratio = (
        df.duplicated().sum() / len(df)
        if len(df) > 0
        else 0
    )

    duplicate_score = 1 - duplicate_ratio

    # ---------------------------------------------------
    # OUTLIER SCORE
    # ---------------------------------------------------

    numeric_df = df.select_dtypes(include=np.number)

    total_outliers = 0
    total_numeric = 0

    for col in numeric_df.columns:

        series = numeric_df[col].dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outliers = (
            (series < lower) |
            (series > upper)
        ).sum()

        total_outliers += outliers
        total_numeric += len(series)

        if outliers > 0:
            warnings.append(
                f"{col}: {outliers} suspected outliers"
            )

    outlier_ratio = (
        total_outliers / total_numeric
        if total_numeric > 0
        else 0
    )

    outlier_score = 1 - min(outlier_ratio, 1)

    # ---------------------------------------------------
    # NULL WARNINGS
    # ---------------------------------------------------

    for col in df.columns:

        null_pct = df[col].isna().mean() * 100

        if null_pct > 10:
            warnings.append(
                f"{col}: {null_pct:.1f}% missing values"
            )

    # ---------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------

    final_score = (
        completeness_score * 0.5
        + duplicate_score * 0.2
        + outlier_score * 0.3
    )

    final_score = round(
        max(0.0, min(final_score, 1.0)),
        3,
    )

    return {
        "score": final_score,
        "warnings": warnings,
    }