import pandas as pd
import os
from sqlalchemy.orm import Session
from app.models.database_models import Creator, engine, Base

def load_and_clean_data(csv_path: str = "yt.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")

    # Handle basic missing values or formatting if necessary
    # For numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # For categorical
    # df['category'] = df['category'].fillna('Unknown')

    # Let's keep NaNs for DB insertion where appropriate, or handle them cleanly.
    # The database models allow Nulls (Optional by default in schema, nullable=True in SQLAlchemy)
    # We will replace NaN with None for SQLAlchemy
    df = df.replace({pd.NA: None, float('nan'): None})
    return df

def ingest_data_to_db(csv_path: str = "yt.csv"):
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    print("Loading data...")
    df = load_and_clean_data(csv_path)

    from app.models.database_models import SessionLocal
    db: Session = SessionLocal()

    try:
        # Check if data already exists
        if db.query(Creator).first():
            print("Data already exists in database. Skipping ingestion.")
            return

        creators = []
        for _, row in df.iterrows():
            creator = Creator(
                rank=row.get('rank'),
                youtuber=row.get('Youtuber'),
                subscribers=row.get('subscribers'),
                video_views=row.get('video views'),
                category=row.get('category'),
                title=row.get('Title'),
                uploads=row.get('uploads'),
                country=row.get('Country'),
                abbreviation=row.get('Abbreviation'),
                channel_type=row.get('channel_type'),
                video_views_rank=row.get('video_views_rank'),
                country_rank=row.get('country_rank'),
                channel_type_rank=row.get('channel_type_rank'),
                video_views_for_the_last_30_days=row.get('video_views_for_the_last_30_days'),
                lowest_monthly_earnings=row.get('lowest_monthly_earnings'),
                highest_monthly_earnings=row.get('highest_monthly_earnings'),
                lowest_yearly_earnings=row.get('lowest_yearly_earnings'),
                highest_yearly_earnings=row.get('highest_yearly_earnings'),
                subscribers_for_last_30_days=row.get('subscribers_for_last_30_days'),
                created_year=row.get('created_year'),
                created_month=row.get('created_month'),
                created_date=row.get('created_date'),
                tertiary_education_enrollment=row.get('Gross tertiary education enrollment (%)'),
                population=row.get('Population'),
                unemployment_rate=row.get('Unemployment rate'),
                urban_population=row.get('Urban_population'),
                latitude=row.get('Latitude'),
                longitude=row.get('Longitude'),
            )
            creators.append(creator)

        print(f"Inserting {len(creators)} records...")
        db.bulk_save_objects(creators)
        db.commit()
        print("Data ingestion complete.")
    except Exception as e:
        db.rollback()
        print(f"Error during ingestion: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # If run directly from the backend directory
    # Ensure correct path to yt.csv
    csv_path = "../yt.csv" if os.path.exists("../yt.csv") else "yt.csv"
    ingest_data_to_db(csv_path)
