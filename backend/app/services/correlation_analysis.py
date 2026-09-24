import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Tuple

def calculate_correlation(data1: List[float], data2: List[float], method: str = "pearson") -> Dict[str, Any]:
    df = pd.DataFrame({"v1": data1, "v2": data2}).dropna()
    if len(df) < 2:
        raise ValueError("Insufficient paired data for correlation analysis.")

    n = len(df)

    if method.lower() == "pearson":
        coef, p_val = stats.pearsonr(df["v1"], df["v2"])
        method_name = "Pearson"
    elif method.lower() == "spearman":
        coef, p_val = stats.spearmanr(df["v1"], df["v2"])
        method_name = "Spearman"
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")

    coef = float(coef)
    p_val = float(p_val)

    # Interpretation
    interpretation = f"{method_name} correlation indicates a "

    if abs(coef) > 0.7: strength = "strong"
    elif abs(coef) > 0.3: strength = "moderate"
    else: strength = "weak"

    direction = "positive" if coef > 0 else "negative"

    if p_val < 0.05:
        interpretation += f"statistically significant, {strength} {direction} relationship."
    else:
        interpretation += f"non-significant relationship (p={p_val:.4f})."

    interpretation += " Note: Correlation does not establish causation."

    return {
        "method": method_name,
        "coefficient": coef,
        "p_value": p_val,
        "sample_size": n,
        "interpretation": interpretation
    }

def calculate_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()
    if method.lower() == "pearson":
        corr_matrix = numeric_df.corr(method="pearson")
    else:
        corr_matrix = numeric_df.corr(method="spearman")

    return {
        "method": method.capitalize(),
        "variables": list(corr_matrix.columns),
        "matrix": corr_matrix.values.tolist()
    }
