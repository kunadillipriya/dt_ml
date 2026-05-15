from dt_ml.data_loader import load_csv

def test_load_csv():
    df = load_csv("validation/test_data/sales_clean.csv")
    assert not df.empty