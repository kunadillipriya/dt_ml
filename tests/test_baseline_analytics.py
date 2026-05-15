from dt_ml.baseline_analytics import compute
import pandas as pd

def test_compute():
    df = pd.DataFrame()
    result = compute(df)
    assert isinstance(result, dict)