import pandas as pd

from dt_ml import compute

df = pd.read_csv("validation/test_data/sales_data.csv")

result = compute(df)

print(result)