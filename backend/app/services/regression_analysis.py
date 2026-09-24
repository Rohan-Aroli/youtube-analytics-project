import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Any

def run_regression(df: pd.DataFrame, dep_var: str, indep_vars: List[str]) -> Dict[str, Any]:
    # Drop rows with NaN in the selected columns
    model_df = df[[dep_var] + indep_vars].dropna()

    if len(model_df) < len(indep_vars) + 2:
        raise ValueError("Insufficient data for regression analysis.")

    y = model_df[dep_var]
    X = model_df[indep_vars]

    # Add constant for intercept
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    # Calculate RMSE and MAE
    predictions = model.predict(X)
    rmse = np.sqrt(np.mean((y - predictions)**2))
    mae = np.mean(np.abs(y - predictions))

    # Coefficients
    coef_list = []
    for var in model.params.index:
        if var != 'const':
            coef_list.append({
                "variable": var,
                "coefficient": float(model.params[var]),
                "p_value": float(model.pvalues[var]),
                "ci_lower": float(model.conf_int().loc[var, 0]),
                "ci_upper": float(model.conf_int().loc[var, 1])
            })

    # Diagnostics - Multicollinearity (VIF)
    vif_data = {}
    if len(indep_vars) > 1:
        # Don't include constant for VIF if possible, or handle correctly. We use the original X matrix without constant
        X_vif = model_df[indep_vars]
        for i, var in enumerate(X_vif.columns):
            try:
                vif = variance_inflation_factor(X_vif.values, i)
                vif_data[var] = float(vif)
            except:
                pass

    diagnostics = {"vif": vif_data}

    equation = f"{dep_var} = {model.params.get('const', 0):.4f}"
    for coef in coef_list:
        sign = "+" if coef['coefficient'] >= 0 else "-"
        equation += f" {sign} {abs(coef['coefficient']):.4f}*{coef['variable']}"

    return {
        "equation": equation,
        "intercept": float(model.params.get('const', 0)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "rmse": float(rmse),
        "mae": float(mae),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": coef_list,
        "diagnostics": diagnostics
    }
