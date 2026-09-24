from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.database_models import get_db, Creator
from app.schemas.analytics import DashboardSummary

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db)):
    total = db.query(Creator).count()

    # Calculate means
    mean_subscribers = db.query(func.avg(Creator.subscribers)).scalar() or 0
    mean_views = db.query(func.avg(Creator.video_views)).scalar() or 0
    mean_earnings = db.query(func.avg(Creator.highest_yearly_earnings)).scalar() or 0

    # Calculate medians (SQLite doesn't have a built-in median function easily accessible via SQLAlchemy without custom dialect work, so we fetch and calculate in Python for simplicity in SQLite, though in Postgres we'd use percentile_cont)
    # Fetch all relevant columns
    data = db.query(Creator.subscribers, Creator.video_views, Creator.highest_yearly_earnings).all()
    subs = sorted([r[0] for r in data if r[0] is not None])
    views = sorted([r[1] for r in data if r[1] is not None])
    earnings = sorted([r[2] for r in data if r[2] is not None])

    def get_median(lst):
        if not lst: return 0
        n = len(lst)
        if n % 2 == 1: return lst[n//2]
        else: return (lst[n//2 - 1] + lst[n//2]) / 2.0

    median_subscribers = get_median(subs)
    median_views = get_median(views)
    median_earnings = get_median(earnings)

    # Counts
    num_categories = db.query(func.count(func.distinct(Creator.category))).scalar() or 0
    num_countries = db.query(func.count(func.distinct(Creator.country))).scalar() or 0

    return DashboardSummary(
        total_creators=total,
        mean_subscribers=mean_subscribers,
        median_subscribers=median_subscribers,
        mean_views=mean_views,
        median_views=median_views,
        mean_earnings=mean_earnings,
        median_earnings=median_earnings,
        num_categories=num_categories,
        num_countries=num_countries
    )
