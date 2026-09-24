from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_analytics.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Creator(Base):
    __tablename__ = "creators"

    id = Column(Integer, primary_key=True, index=True)
    rank = Column(Integer)
    youtuber = Column(String, index=True)
    subscribers = Column(Float)
    video_views = Column(Float)
    category = Column(String, index=True)
    title = Column(String)
    uploads = Column(Integer)
    country = Column(String, index=True)
    abbreviation = Column(String)
    channel_type = Column(String)
    video_views_rank = Column(Float)
    country_rank = Column(Float)
    channel_type_rank = Column(Float)
    video_views_for_the_last_30_days = Column(Float)
    lowest_monthly_earnings = Column(Float)
    highest_monthly_earnings = Column(Float)
    lowest_yearly_earnings = Column(Float)
    highest_yearly_earnings = Column(Float)
    subscribers_for_last_30_days = Column(Float)
    created_year = Column(Float)
    created_month = Column(String)
    created_date = Column(Float)
    tertiary_education_enrollment = Column(Float)
    population = Column(Float)
    unemployment_rate = Column(Float)
    urban_population = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
