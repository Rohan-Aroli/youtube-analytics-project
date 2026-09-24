from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.models.database_models import get_db, Creator
from app.schemas.statistics import DescriptiveStats
from app.services.descriptive_stats import calculate_descriptive_stats

router = APIRouter(prefix="/api/statistics", tags=["Statistics"])

@router.get("/descriptive", response_model=DescriptiveStats)
def get_descriptive_statistics(
    variable: str = Query(..., description="Numerical variable to analyze (e.g., subscribers, video_views, uploads, highest_yearly_earnings)"),
    db: Session = Depends(get_db)
):
    valid_columns = {
        "subscribers": Creator.subscribers,
        "video_views": Creator.video_views,
        "uploads": Creator.uploads,
        "highest_yearly_earnings": Creator.highest_yearly_earnings,
        "lowest_yearly_earnings": Creator.lowest_yearly_earnings,
        "highest_monthly_earnings": Creator.highest_monthly_earnings,
        "lowest_monthly_earnings": Creator.lowest_monthly_earnings,
        "video_views_for_the_last_30_days": Creator.video_views_for_the_last_30_days,
        "subscribers_for_last_30_days": Creator.subscribers_for_last_30_days
    }

    if variable not in valid_columns:
        raise HTTPException(status_code=400, detail=f"Invalid variable. Allowed variables: {list(valid_columns.keys())}")

    data = [r[0] for r in db.query(valid_columns[variable]).all()]
    try:
        stats_result = calculate_descriptive_stats(data)
        return stats_result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

from app.schemas.statistics import DistributionStats
from app.services.distribution_analysis import analyze_distribution

@router.get("/distribution", response_model=DistributionStats)
def get_distribution_analysis(
    variable: str = Query(..., description="Numerical variable to analyze"),
    db: Session = Depends(get_db)
):
    valid_columns = {
        "subscribers": Creator.subscribers,
        "video_views": Creator.video_views,
        "uploads": Creator.uploads,
        "highest_yearly_earnings": Creator.highest_yearly_earnings,
        "lowest_yearly_earnings": Creator.lowest_yearly_earnings,
        "highest_monthly_earnings": Creator.highest_monthly_earnings,
        "lowest_monthly_earnings": Creator.lowest_monthly_earnings,
        "video_views_for_the_last_30_days": Creator.video_views_for_the_last_30_days,
        "subscribers_for_last_30_days": Creator.subscribers_for_last_30_days
    }

    if variable not in valid_columns:
        raise HTTPException(status_code=400, detail=f"Invalid variable. Allowed variables: {list(valid_columns.keys())}")

    data = [r[0] for r in db.query(valid_columns[variable]).all()]
    try:
        dist_result = analyze_distribution(data, variable_name=variable)
        return dist_result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
