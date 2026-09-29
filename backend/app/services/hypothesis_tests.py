import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any

def run_one_sample_ttest(data: List[float], popmean: float, alpha: float = 0.05, variable: str = "Variable") -> Dict[str, Any]:
    series = pd.Series([x for x in data if x is not None]).dropna()
    if len(series) < 2:
        raise ValueError("Insufficient data for 1-sample t-test.")

    t_stat, p_val = stats.ttest_1samp(series, popmean)
    mean_val = series.mean()

    # Cohen's d
    d = (mean_val - popmean) / series.std()

    # Confidence Interval
    ci = stats.t.interval(1 - alpha, len(series)-1, loc=mean_val, scale=stats.sem(series))

    reject_null = p_val < alpha

    interpretation = f"We {'reject' if reject_null else 'fail to reject'} the null hypothesis. "
    if reject_null:
        interpretation += f"There is statistically significant evidence that the true mean of {variable} differs from {popmean}. "
    else:
        interpretation += f"There is not enough evidence to conclude the true mean of {variable} differs from {popmean}. "

    return {
        "test_name": "One-sample t-test",
        "null_hypothesis": f"The population mean of {variable} equals {popmean}.",
        "alternative_hypothesis": f"The population mean of {variable} does not equal {popmean}.",
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "alpha": alpha,
        "decision": "Reject Null" if reject_null else "Fail to Reject Null",
        "interpretation": interpretation,
        "effect_size": float(d),
        "confidence_interval": (float(ci[0]), float(ci[1])),
        "additional_info": {"sample_mean": float(mean_val), "sample_size": len(series)}
    }

def run_independent_ttest(data1: List[float], data2: List[float], group1_name: str, group2_name: str, variable: str, alpha: float = 0.05) -> Dict[str, Any]:
    s1 = pd.Series([x for x in data1 if x is not None]).dropna()
    s2 = pd.Series([x for x in data2 if x is not None]).dropna()

    if len(s1) < 2 or len(s2) < 2:
        raise ValueError("Insufficient data in groups for independent t-test.")

    t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=False) # Welch's t-test by default

    mean1, mean2 = s1.mean(), s2.mean()
    mean_diff = mean1 - mean2

    # Cohen's d for independent samples
    pooled_std = np.sqrt(((len(s1)-1)*s1.var() + (len(s2)-1)*s2.var()) / (len(s1)+len(s2)-2))
    d = mean_diff / pooled_std if pooled_std > 0 else 0

    reject_null = p_val < alpha

    interpretation = f"We {'reject' if reject_null else 'fail to reject'} the null hypothesis. "
    if reject_null:
        interpretation += f"There is statistically significant evidence of a difference in mean {variable} between {group1_name} and {group2_name}. "
    else:
        interpretation += f"There is not enough evidence to conclude a difference in mean {variable} between the groups. "

    return {
        "test_name": "Independent Two-sample t-test (Welch's)",
        "null_hypothesis": f"Mean {variable} for {group1_name} equals mean {variable} for {group2_name}.",
        "alternative_hypothesis": f"Means are not equal.",
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "alpha": alpha,
        "decision": "Reject Null" if reject_null else "Fail to Reject Null",
        "interpretation": interpretation,
        "effect_size": float(d),
        "additional_info": {
            "group1_mean": float(mean1), "group1_size": len(s1),
            "group2_mean": float(mean2), "group2_size": len(s2),
            "mean_difference": float(mean_diff)
        }
    }

def run_anova(df: pd.DataFrame, num_col: str, cat_col: str, alpha: float = 0.05) -> Dict[str, Any]:
    df_clean = df[[num_col, cat_col]].dropna()
    groups = df_clean.groupby(cat_col)[num_col].apply(list)

    if len(groups) < 2:
        raise ValueError("Need at least 2 distinct groups for ANOVA.")

    f_stat, p_val = stats.f_oneway(*groups)
    reject_null = p_val < alpha

    interpretation = f"We {'reject' if reject_null else 'fail to reject'} the null hypothesis. "
    if reject_null:
        interpretation += f"There is statistically significant evidence that mean {num_col} differs across {cat_col} groups. "
        # Post-hoc could be added here (e.g., Tukey HSD using statsmodels), but omitted for brevity unless needed.
    else:
        interpretation += f"There is not enough evidence to conclude mean {num_col} differs across {cat_col} groups. "

    group_means = df_clean.groupby(cat_col)[num_col].mean().to_dict()

    return {
        "test_name": "One-way ANOVA",
        "null_hypothesis": f"All group means of {num_col} across {cat_col} are equal.",
        "alternative_hypothesis": f"At least one group mean is different.",
        "statistic": float(f_stat),
        "p_value": float(p_val),
        "alpha": alpha,
        "decision": "Reject Null" if reject_null else "Fail to Reject Null",
        "interpretation": interpretation,
        "additional_info": {"group_means": {k: float(v) for k, v in group_means.items()}}
    }

def run_chi_square(df: pd.DataFrame, cat1: str, cat2: str, alpha: float = 0.05) -> Dict[str, Any]:
    df_clean = df[[cat1, cat2]].dropna()
    contingency = pd.crosstab(df_clean[cat1], df_clean[cat2])

    if contingency.size < 4:
        raise ValueError("Insufficient categories for Chi-square.")

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    reject_null = p_val < alpha

    interpretation = f"We {'reject' if reject_null else 'fail to reject'} the null hypothesis. "
    if reject_null:
        interpretation += f"There is statistically significant evidence of an association between {cat1} and {cat2}. "
    else:
        interpretation += f"There is not enough evidence to conclude an association between {cat1} and {cat2}. "

    # Cramer's V
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return {
        "test_name": "Chi-square Test of Independence",
        "null_hypothesis": f"{cat1} and {cat2} are independent.",
        "alternative_hypothesis": f"{cat1} and {cat2} are associated.",
        "statistic": float(chi2),
        "p_value": float(p_val),
        "alpha": alpha,
        "decision": "Reject Null" if reject_null else "Fail to Reject Null",
        "interpretation": interpretation,
        "effect_size": float(v), # Cramer's V
        "additional_info": {"degrees_of_freedom": int(dof), "cramers_v": float(v)}
    }
