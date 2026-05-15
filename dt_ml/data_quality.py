# dt_ml/data_quality.py

def quality_score(df):
    missing_ratio = df.isnull().mean().mean()

    score = 1 - missing_ratio

    return round(score, 2)