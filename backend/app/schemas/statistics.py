from pydantic import BaseModel
from typing import Optional, Dict, Any

class DescriptiveStats(BaseModel):
    count: int
    mean: float
    median: float
    mode: Optional[float] = None
    min: float
    max: float
    range: float
    variance: float
    std_dev: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    interpretation: str

class DistributionStats(BaseModel):
    skewness: float
    kurtosis: float
    shapiro_wilk_stat: Optional[float] = None
    shapiro_wilk_p: Optional[float] = None
    is_normal: bool
    interpretation: str
    histogram_data: list
    boxplot_data: list

class CorrelationResult(BaseModel):
    method: str
    coefficient: float
    p_value: float
    sample_size: int
    interpretation: str

class CorrelationMatrixResponse(BaseModel):
    method: str
    variables: list[str]
    matrix: list[list[float]]
