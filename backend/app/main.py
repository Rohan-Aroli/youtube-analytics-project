from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import dashboard, creators, statistics, correlation, hypothesis, regression, ml

app = FastAPI(
    title="YouTube Creator Analytics API",
    description="Statistical Intelligence Platform for YouTube Creators Data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(creators.router)
app.include_router(statistics.router)
app.include_router(correlation.router)
app.include_router(hypothesis.router)
app.include_router(regression.router)
app.include_router(ml.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to YouTube Creator Analytics API"}
