import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any

def analyze_distribution(data: List[float], variable_name: str = "variable") -> Dict[str, Any]:
    series = pd.Series([x for x in data if x is not None]).dropna()
    if len(series) < 3: # Need at least 3 points for Shapiro-Wilk
        raise ValueError("Insufficient data for distribution analysis. Need at least 3 valid points.")

    skewness = float(series.skew())
    kurtosis = float(series.kurt())

    # Shapiro-Wilk test for normality
    # It might be inaccurate for N > 5000, and N > 50 tends to easily reject null hypothesis.
    # For N > 5000 scipy raises a warning and p-value may not be accurate.
    # Our dataset is ~1000, so it's fine.
    shapiro_stat, shapiro_p = stats.shapiro(series)
    shapiro_stat = float(shapiro_stat)
    shapiro_p = float(shapiro_p)

    is_normal = shapiro_p > 0.05

    # Interpretation
    interpretation = f"The Shapiro-Wilk test yielded a p-value of {shapiro_p:.4f}. "
    if is_normal:
        interpretation += "We fail to reject the null hypothesis, suggesting the distribution is approximately normal. "
    else:
        interpretation += "We reject the null hypothesis, indicating the distribution significantly deviates from normality. "
        if len(series) > 300:
            interpretation += f"Note: With a large sample size (n={len(series)}), even minor deviations from normality can become statistically significant. Visually inspect the histogram and Q-Q plot. "

    if skewness > 1:
        interpretation += f"The {variable_name} distribution is strongly right-skewed. "
    elif skewness < -1:
        interpretation += f"The {variable_name} distribution is strongly left-skewed. "

    # Generate histogram data for frontend charts (bins)
    counts, bin_edges = np.histogram(series, bins='auto')
    histogram_data = [{"bin_start": float(bin_edges[i]), "bin_end": float(bin_edges[i+1]), "count": int(counts[i])} for i in range(len(counts))]

    # Generate boxplot data (five number summary plus outliers)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()

    boxplot_data = [{
        "min": float(series.min()),
        "q1": float(q1),
        "median": float(series.median()),
        "q3": float(q3),
        "max": float(series.max()),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outliers": [float(x) for x in outliers]
    }]

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "shapiro_wilk_stat": shapiro_stat,
        "shapiro_wilk_p": shapiro_p,
        "is_normal": is_normal,
        "interpretation": interpretation.strip(),
        "histogram_data": histogram_data,
        "boxplot_data": boxplot_data
    }
