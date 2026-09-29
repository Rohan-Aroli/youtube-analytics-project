from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.models.database_models import get_db, Creator
from app.schemas.statistics import CorrelationResult, CorrelationMatrixResponse
from app.services.correlation_analysis import calculate_correlation, calculate_correlation_matrix
import pandas as pd

router = APIRouter(prefix="/api/statistics", tags=["Correlation"])

@router.get("/correlation", response_model=CorrelationResult)
def get_correlation(
    var1: str = Query(..., description="First variable"),
    var2: str = Query(..., description="Second variable"),
    method: str = Query("pearson", description="Method: 'pearson' or 'spearman'"),
    db: Session = Depends(get_db)
):
    valid_columns = {
        "subscribers": Creator.subscribers,
        "video_views": Creator.video_views,
        "uploads": Creator.uploads,
        "highest_yearly_earnings": Creator.highest_yearly_earnings,
        "lowest_yearly_earnings": Creator.lowest_yearly_earnings,
        "video_views_for_the_last_30_days": Creator.video_views_for_the_last_30_days,
        "subscribers_for_last_30_days": Creator.subscribers_for_last_30_days,
        "population": Creator.population,
        "unemployment_rate": Creator.unemployment_rate
    }

    if var1 not in valid_columns or var2 not in valid_columns:
        raise HTTPException(status_code=400, detail="Invalid variable(s) provided.")

    data = db.query(valid_columns[var1], valid_columns[var2]).all()
    data1 = [r[0] for r in data]
    data2 = [r[1] for r in data]

    try:
        result = calculate_correlation(data1, data2, method=method)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/correlation/matrix", response_model=CorrelationMatrixResponse)
def get_correlation_matrix(
    method: str = Query("pearson", description="Method: 'pearson' or 'spearman'"),
    db: Session = Depends(get_db)
):
    # Fetch relevant columns
    data = db.query(
        Creator.subscribers,
        Creator.video_views,
        Creator.uploads,
        Creator.highest_yearly_earnings,
        Creator.video_views_for_the_last_30_days
    ).all()

    df = pd.DataFrame(data, columns=["subscribers", "video_views", "uploads", "highest_yearly_earnings", "views_last_30d"])

    try:
        result = calculate_correlation_matrix(df, method=method)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
