from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.database_models import get_db, Creator
from app.schemas.hypothesis import HypothesisTestResult, TTestRequest, ANOVARequest, ChiSquareRequest
from app.services.hypothesis_tests import run_one_sample_ttest, run_independent_ttest, run_anova, run_chi_square
import pandas as pd

router = APIRouter(prefix="/api/statistics/hypothesis", tags=["Hypothesis Testing"])

def get_column(db: Session, var_name: str):
    valid_columns = {
        "subscribers": Creator.subscribers,
        "video_views": Creator.video_views,
        "uploads": Creator.uploads,
        "highest_yearly_earnings": Creator.highest_yearly_earnings,
        "category": Creator.category,
        "country": Creator.country,
        "channel_type": Creator.channel_type
    }
    if var_name not in valid_columns:
        raise HTTPException(status_code=400, detail=f"Invalid variable: {var_name}")
    return valid_columns[var_name]

@router.post("/t-test", response_model=HypothesisTestResult)
def perform_ttest(req: TTestRequest, db: Session = Depends(get_db)):
    col = get_column(db, req.variable)

    if req.group_by and req.group1_value and req.group2_value:
        # Independent t-test
        group_col = get_column(db, req.group_by)
        data1 = [r[0] for r in db.query(col).filter(group_col == req.group1_value).all()]
        data2 = [r[0] for r in db.query(col).filter(group_col == req.group2_value).all()]
        try:
            return run_independent_ttest(data1, data2, req.group1_value, req.group2_value, req.variable, req.alpha)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif req.popmean is not None:
        # 1-sample
        data = [r[0] for r in db.query(col).all()]
        try:
            return run_one_sample_ttest(data, req.popmean, req.alpha, req.variable)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Must provide either popmean for 1-sample or group_by details for independent t-test.")

@router.post("/anova", response_model=HypothesisTestResult)
def perform_anova(req: ANOVARequest, db: Session = Depends(get_db)):
    num_col = get_column(db, req.numerical_variable)
    cat_col = get_column(db, req.categorical_variable)

    data = db.query(num_col, cat_col).all()
    df = pd.DataFrame(data, columns=[req.numerical_variable, req.categorical_variable])
    try:
        return run_anova(df, req.numerical_variable, req.categorical_variable, req.alpha)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chi-square", response_model=HypothesisTestResult)
def perform_chisquare(req: ChiSquareRequest, db: Session = Depends(get_db)):
    cat1_col = get_column(db, req.variable1)
    cat2_col = get_column(db, req.variable2)

    data = db.query(cat1_col, cat2_col).all()
    df = pd.DataFrame(data, columns=[req.variable1, req.variable2])
    try:
        return run_chi_square(df, req.variable1, req.variable2, req.alpha)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
