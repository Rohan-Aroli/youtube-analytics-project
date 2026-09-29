from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
from app.schemas.ml import MLModelMetrics, MLPredictionRequest, MLPredictionResponse
from app.services.ml_service import train_earnings_models, train_classification_models, get_model

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

# In a real app, training wouldn't be done on every restart like this.
# We'll just load or train them here for simplicity if not already trained
try:
    earnings_results = train_earnings_models()
    class_results = train_classification_models()
except Exception as e:
    print("Warning: Failed to pre-train ML models:", e)
    earnings_results = {}
    class_results = {}

@router.get("/models", response_model=List[MLModelMetrics])
def get_ml_models():
    models_info = []
    for name, res in earnings_results.items():
        models_info.append(MLModelMetrics(
            model_name=name,
            task_type="regression",
            metrics=res["metrics"],
            feature_importance=res["feature_importance"]
        ))
    for name, res in class_results.items():
        models_info.append(MLModelMetrics(
            model_name=name,
            task_type="classification",
            metrics=res["metrics"],
            feature_importance=res["feature_importance"]
        ))
    return models_info

@router.post("/predict", response_model=MLPredictionResponse)
def predict_earnings(req: MLPredictionRequest):
    model = get_model(req.model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    df = pd.DataFrame([req.features])
    try:
        pred = model.predict(df)[0]
        return MLPredictionResponse(prediction=float(pred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/classify", response_model=MLPredictionResponse)
def classify_success(req: MLPredictionRequest):
    model = get_model(req.model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    df = pd.DataFrame([req.features])
    try:
        pred = model.predict(df)[0]
        prob = None
        if hasattr(model.named_steps['model'], "predict_proba"):
            prob = model.predict_proba(df)[0][1]

        return MLPredictionResponse(
            prediction=float(pred),
            predicted_class="Successful" if pred == 1 else "Not Successful",
            probability=float(prob) if prob is not None else None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
