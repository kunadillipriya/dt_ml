# tests/test_edge_cases.py

from dt_ml.data_loader import load_csv


def test_duplicate_removal():

    df = load_csv("validation/test_data/sales_data.csv")

    assert df.duplicated().sum() == 0


def test_date_parsing():

    df = load_csv("validation/test_data/sales_data.csv")

    date_cols = [c for c in df.columns if "date" in c]

    assert len(date_cols) > 0


def test_column_normalization():

    df = load_csv("validation/test_data/sales_data.csv")

    for col in df.columns:
        assert " " not in col


def test_numeric_cleaning():

    df = load_csv("validation/test_data/sales_data.csv")

    numeric_cols = df.select_dtypes(include="number").columns

    assert len(numeric_cols) > 0