import cProfile 
import pstats 
import pandas as pd 
from dt_ml import compute 
df = pd.read_csv("validation/test_data/business_data_large.csv") 
profiler = cProfile.Profile() 
profiler.enable() 
result = compute(df)
profiler.disable() 
stats = pstats.Stats(profiler) 
stats.sort_stats("cumtime")
stats.print_stats(25)