from pydantic import BaseModel
from typing import Optional, List

class CreatorBase(BaseModel):
    rank: Optional[int] = None
    youtuber: Optional[str] = None
    subscribers: Optional[float] = None
    video_views: Optional[float] = None
    category: Optional[str] = None
    title: Optional[str] = None
    uploads: Optional[int] = None
    country: Optional[str] = None
    abbreviation: Optional[str] = None
    channel_type: Optional[str] = None
    video_views_rank: Optional[float] = None
    country_rank: Optional[float] = None
    channel_type_rank: Optional[float] = None
    video_views_for_the_last_30_days: Optional[float] = None
    lowest_monthly_earnings: Optional[float] = None
    highest_monthly_earnings: Optional[float] = None
    lowest_yearly_earnings: Optional[float] = None
    highest_yearly_earnings: Optional[float] = None
    subscribers_for_last_30_days: Optional[float] = None
    created_year: Optional[float] = None
    created_month: Optional[str] = None
    created_date: Optional[float] = None

class CreatorSchema(CreatorBase):
    id: int
    class Config:
        from_attributes = True

class DashboardSummary(BaseModel):
    total_creators: int
    mean_subscribers: float
    median_subscribers: float
    mean_views: float
    median_views: float
    mean_earnings: float
    median_earnings: float
    num_categories: int
    num_countries: int

class CreatorComparisonResponse(BaseModel):
    creators: List[CreatorSchema]
    comparison_metrics: dict
