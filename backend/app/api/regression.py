from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.database_models import get_db, Creator
from app.schemas.regression import RegressionResult, RegressionRequest
from app.services.regression_analysis import run_regression
import pandas as pd

router = APIRouter(prefix="/api/statistics/regression", tags=["Regression Analysis"])

def get_column(db: Session, var_name: str):
    valid_columns = {
        "subscribers": Creator.subscribers,
        "video_views": Creator.video_views,
        "uploads": Creator.uploads,
        "highest_yearly_earnings": Creator.highest_yearly_earnings,
        "lowest_yearly_earnings": Creator.lowest_yearly_earnings,
        "highest_monthly_earnings": Creator.highest_monthly_earnings,
        "lowest_monthly_earnings": Creator.lowest_monthly_earnings,
        "video_views_for_the_last_30_days": Creator.video_views_for_the_last_30_days,
        "subscribers_for_last_30_days": Creator.subscribers_for_last_30_days,
        "population": Creator.population,
        "unemployment_rate": Creator.unemployment_rate
    }
    if var_name not in valid_columns:
        raise HTTPException(status_code=400, detail=f"Invalid variable: {var_name}")
    return valid_columns[var_name]

@router.post("", response_model=RegressionResult)
def perform_regression(req: RegressionRequest, db: Session = Depends(get_db)):
    if not req.independent_variables:
        raise HTTPException(status_code=400, detail="Must provide at least one independent variable.")

    cols = [get_column(db, req.dependent_variable)] + [get_column(db, var) for var in req.independent_variables]

    data = db.query(*cols).all()
    col_names = [req.dependent_variable] + req.independent_variables
    df = pd.DataFrame(data, columns=col_names)

    try:
        return run_regression(df, req.dependent_variable, req.independent_variables)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression failed: {str(e)}")
