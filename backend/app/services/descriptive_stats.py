import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any

def calculate_descriptive_stats(data: List[float]) -> Dict[str, Any]:
    series = pd.Series([x for x in data if x is not None]).dropna()
    if len(series) == 0:
        raise ValueError("No valid data provided for descriptive statistics.")

    count = int(series.count())
    mean = float(series.mean())
    median = float(series.median())
    try:
        mode = float(series.mode().iloc[0])
    except:
        mode = None

    min_val = float(series.min())
    max_val = float(series.max())
    range_val = max_val - min_val
    variance = float(series.var()) if count > 1 else 0.0
    std_dev = float(series.std()) if count > 1 else 0.0

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    skewness = float(series.skew()) if count > 2 else 0.0
    kurtosis = float(series.kurt()) if count > 3 else 0.0

    # Generate Interpretation
    interpretation = ""
    if skewness > 1:
        interpretation += "The distribution is strongly right-skewed, indicating a tail extending to higher values. "
    elif skewness < -1:
        interpretation += "The distribution is strongly left-skewed, indicating a tail extending to lower values. "
    elif -0.5 <= skewness <= 0.5:
        interpretation += "The distribution is approximately symmetric. "
    else:
        interpretation += "The distribution is moderately skewed. "

    if kurtosis > 3:
        interpretation += "It is leptokurtic, meaning it has heavy tails and outliers are more likely."
    elif kurtosis < -1:
        interpretation += "It is platykurtic, meaning it has lighter tails."
    else:
        interpretation += "It has roughly normal kurtosis (mesokurtic)."

    return {
        "count": count,
        "mean": mean,
        "median": median,
        "mode": mode,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "variance": variance,
        "std_dev": std_dev,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "interpretation": interpretation.strip()
    }
