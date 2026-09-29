from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class MLModelMetrics(BaseModel):
    model_name: str
    task_type: str
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None

class MLPredictionRequest(BaseModel):
    model_name: str
    features: Dict[str, Any]

class MLPredictionResponse(BaseModel):
    prediction: float
    prediction_interval: Optional[tuple[float, float]] = None
    probability: Optional[float] = None
    predicted_class: Optional[str] = None
