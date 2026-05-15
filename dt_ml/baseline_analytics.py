import pandas as pd
def compute(df: pd.DataFrame):
    """
    Baseline analytics function.
    """
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }
    return summary