from concurrent.futures import ThreadPoolExecutor 
import pandas as pd 
from dt_ml import compute 
df = pd.read_csv( "validation/test_data/business_data_large.csv" ) 
def run_once(): 
    return compute(df)
with ThreadPoolExecutor(max_workers=20) as executor:
     futures = [ executor.submit(run_once) for _ in range(100) ] 
results = [f.result() for f in futures] 
print("Completed:", len(results))