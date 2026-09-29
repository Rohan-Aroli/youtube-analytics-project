from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class HypothesisTestResult(BaseModel):
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    decision: str
    interpretation: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    additional_info: Optional[Dict[str, Any]] = None

class TTestRequest(BaseModel):
    variable: str
    group_by: Optional[str] = None
    group1_value: Optional[str] = None
    group2_value: Optional[str] = None
    popmean: Optional[float] = None
    alpha: float = 0.05

class ANOVARequest(BaseModel):
    numerical_variable: str
    categorical_variable: str
    alpha: float = 0.05

class ChiSquareRequest(BaseModel):
    variable1: str
    variable2: str
    alpha: float = 0.05
