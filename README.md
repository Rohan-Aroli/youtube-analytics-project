# 📊 YouTube Creator Analytics & Statistical Intelligence Platform

A comprehensive full-stack analytics platform enabling deep statistical analysis and machine learning predictions on YouTube creator data. This project transforms a basic EDA script into a robust web application built with FastAPI, PostgreSQL/SQLite, React, and Recharts.

---

## 🔗 Project Overview

This platform takes raw YouTube creator metrics and allows users to explore descriptive statistics, probability distributions, correlations, hypothesis tests, and machine learning models in an intuitive, academic-grade web interface.

### 🌟 Key Features
- **Dashboard & Dataset Explorer:** High-level metrics and interactive data tables.
- **Descriptive Statistics:** Calculate mean, median, standard deviation, skewness, and kurtosis.
- **Distribution Analysis:** Assess normality visually (Histograms, Box Plots) and statistically (Shapiro-Wilk test).
- **Correlation Analysis:** Pearson and Spearman correlation matrices and significance testing.
- **Hypothesis Testing Lab:** Perform Independent T-Tests, One-way ANOVA, and Chi-Square tests.
- **Regression Analysis:** Simple and Multiple Linear Regression with VIF multicollinearity diagnostics.
- **Machine Learning:** Predict earnings (Random Forest Regressor) and classify creator success (Random Forest Classifier).
- **Creator Comparison:** Visually compare up to 4 creators side-by-side.
- **Executive Statistical Report:** Comprehensive summary of platform findings.

---

## 🛠 Technology Stack

**Backend:**
- Python 3.12+
- FastAPI & Pydantic
- SQLAlchemy (SQLite default, PostgreSQL ready)
- Pandas, NumPy, SciPy, Statsmodels, Scikit-Learn

**Frontend:**
- React 19 (TypeScript)
- Vite
- Tailwind CSS
- Recharts
- Axios & React Router

---

## 📁 Project Structure

```text
youtube-analytics-project/
│── backend/
│   ├── app/
│   │   ├── api/          # FastAPI Route handlers
│   │   ├── services/     # Statistical and ML logic, Data ingestion
│   │   ├── schemas/      # Pydantic models for validation
│   │   ├── models/       # SQLAlchemy database models
│   │   └── main.py       # FastAPI application entry point
│   ├── tests/            # Pytest test cases
│   └── models_cache/     # Saved Scikit-Learn pipelines (.pkl)
│
│── frontend/
│   ├── src/
│   │   ├── components/   # Reusable charts, tables, cards, layout
│   │   ├── pages/        # Dashboard, Reports, Statistics, ML pages
│   │   ├── App.tsx       # React Router setup
│   │   └── main.tsx      # React entry point
│   ├── tailwind.config.js
│   └── package.json
│
│── yt.csv                # Original YouTube dataset
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Backend Setup

1. **Navigate to the root directory and activate a virtual environment:**
   ```
   source venv/bin/activate
   ```

2. **Install backend dependencies:**
   ```
   pip install fastapi uvicorn sqlalchemy pydantic pandas numpy scipy statsmodels scikit-learn python-multipart
   ```

3. **Initialize the Database and Machine Learning Models:**
   ```
   # Load data into SQLite
   export PYTHONPATH=$(pwd)/backend
   python backend/app/services/data_service.py

   # Train and save ML models locally
   python -c "from app.services.ml_service import train_earnings_models, train_classification_models; train_earnings_models(); train_classification_models()"
   ```

4. **Run the FastAPI Server:**
   ```
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   The backend API will be available at `http://localhost:8000`.

### 2. Frontend Setup

1. **Open a new terminal and navigate to the frontend directory:**
   ```
   cd frontend
   ```

2. **Install Node dependencies:**
   ```
   npm install
   ```

3. **Start the Development Server:**
   ```
   npm run dev &
   ```
   The React application will be available at `http://localhost:5173`.

---

## 🗄️ Database Setup (Optional: PostgreSQL)

By default, the application uses an SQLite database (`youtube_analytics.db`) which is perfectly suited for local development.

To use PostgreSQL instead, create a `.env` file in the root directory and specify the connection string:

```
DATABASE_URL=postgresql://user:password@localhost:5432/youtube_analytics
```
*Note: Ensure `psycopg2-binary` is installed via pip if using PostgreSQL.*

---

## 🧪 Running Tests

**Backend:**
```
cd backend
source ../venv/bin/activate
export PYTHONPATH=$(pwd)
pytest tests/
```

**Frontend Build Test:**
```
cd frontend
npm run build
```

---

## 🔬 Statistical & ML Methodology

- **Normality:** Shapiro-Wilk test is used. Given N ≈ 1000, interpretation warns of high sensitivity to minor deviations.
- **Correlations:** Users can toggle between Pearson (linear) and Spearman (monotonic) rank correlations.
- **Regression Diagnostics:** Includes automated Variance Inflation Factor (VIF) calculation to warn against multicollinearity.
- **Machine Learning Preprocessing:** Pipeline utilizes `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`. Random Forests are employed due to robust performance on non-linear interaction features in this dataset. Target leakage is carefully handled (e.g., removing subscriber counts when predicting classification success based on top quartile subscriber metrics).
