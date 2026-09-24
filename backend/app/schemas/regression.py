from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class RegressionCoefficient(BaseModel):
    variable: str
    coefficient: float
    p_value: float
    ci_lower: float
    ci_upper: float

class RegressionResult(BaseModel):
    equation: str
    intercept: float
    r_squared: float
    adj_r_squared: float
    rmse: float
    mae: float
    f_statistic: float
    f_pvalue: float
    coefficients: List[RegressionCoefficient]
    diagnostics: Dict[str, Any]

class RegressionRequest(BaseModel):
    dependent_variable: str
    independent_variables: List[str]
