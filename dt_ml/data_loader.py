# dt_ml/data_loader.py

import pandas as pd

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8")

    # convert dates automatically
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col])

    return df