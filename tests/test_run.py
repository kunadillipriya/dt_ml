import pandas as pd

from dt_ml import compute

df = pd.read_csv("validation/test_data/sales_data.csv")

result = compute(df)

print("\n",result)
from dt_ml import compute_data_quality
a= compute_data_quality(df)
print("Data quality:\n",a)