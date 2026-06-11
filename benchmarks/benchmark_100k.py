import time
import pandas as pd 
from dt_ml import compute 
df = pd.read_csv("validation/test_data/business_data_large.csv") 
start = time.time() 
result = compute(df) 
elapsed = time.time() - start
print("Runtime:", elapsed) 
print("Quality:", result["data_quality_score"])