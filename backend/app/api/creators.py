from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.database_models import get_db, Creator
from app.schemas.analytics import CreatorSchema, CreatorComparisonResponse

router = APIRouter(prefix="/api/creators", tags=["Creators"])

@router.get("", response_model=List[CreatorSchema])
def get_creators(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    country: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Creator)
    if country:
        query = query.filter(Creator.country == country)
    if category:
        query = query.filter(Creator.category == category)

    creators = query.offset(skip).limit(limit).all()
    return creators

@router.get("/compare", response_model=CreatorComparisonResponse)
def compare_creators(
    creator_ids: List[int] = Query(...),
    db: Session = Depends(get_db)
):
    if len(creator_ids) < 2 or len(creator_ids) > 4:
        raise HTTPException(status_code=400, detail="Must select between 2 and 4 creators for comparison.")

    creators = db.query(Creator).filter(Creator.id.in_(creator_ids)).all()
    if not creators:
        raise HTTPException(status_code=404, detail="Creators not found.")

    # Basic comparison metrics setup
    comparison_metrics = {
        "subscribers": {c.youtuber: c.subscribers for c in creators},
        "video_views": {c.youtuber: c.video_views for c in creators},
        "highest_yearly_earnings": {c.youtuber: c.highest_yearly_earnings for c in creators},
        "uploads": {c.youtuber: c.uploads for c in creators}
    }

    return CreatorComparisonResponse(
        creators=creators,
        comparison_metrics=comparison_metrics
    )
