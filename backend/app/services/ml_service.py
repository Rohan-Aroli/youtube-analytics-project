import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from app.models.database_models import SessionLocal, Creator
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_cache")
os.makedirs(MODEL_DIR, exist_ok=True)

def get_ml_dataset(task: str = "earnings"):
    db = SessionLocal()
    query = db.query(
        Creator.subscribers,
        Creator.video_views,
        Creator.uploads,
        Creator.category,
        Creator.country,
        Creator.highest_yearly_earnings,
        Creator.video_views_for_the_last_30_days,
        Creator.subscribers_for_last_30_days
    )
    df = pd.read_sql(query.statement, db.bind)
    db.close()

    # Preprocessing
    df = df.replace({pd.NA: np.nan})
    # Drop rows without target
    if task == "earnings":
        df = df.dropna(subset=['highest_yearly_earnings'])
    elif task == "classification":
        # Success definition: Top 25% of subscribers
        df = df.dropna(subset=['subscribers'])
        threshold = df['subscribers'].quantile(0.75)
        df['is_successful'] = (df['subscribers'] >= threshold).astype(int)

    return df

def build_preprocessor():
    numeric_features = ['subscribers', 'video_views', 'uploads', 'video_views_for_the_last_30_days', 'subscribers_for_last_30_days']
    categorical_features = ['category', 'country']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor, numeric_features, categorical_features


def train_earnings_models():
    df = get_ml_dataset(task="earnings")
    preprocessor, num_cols, cat_cols = build_preprocessor()

    # exclude target from features
    X = df[num_cols + cat_cols]
    y = df['highest_yearly_earnings']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred))
        }

        # Save model
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl"))

        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            # try to extract feature names
            try:
                cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_names = cat_encoder.get_feature_names_out(cat_cols)
                feature_names = num_cols + list(cat_names)
                importances = model.feature_importances_

                # Top 10
                fi_dict = dict(zip(feature_names, importances))
                feature_importance = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)[:10])
            except:
                pass

        results[name] = {"metrics": metrics, "feature_importance": feature_importance}

    return results

def train_classification_models():
    df = get_ml_dataset(task="classification")
    preprocessor, num_cols, cat_cols = build_preprocessor()

    # Remove subscribers from num_cols to avoid leakage!
    leakage_cols = ['subscribers', 'subscribers_for_last_30_days']
    safe_num_cols = [c for c in num_cols if c not in leakage_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), safe_num_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])

    X = df[safe_num_cols + cat_cols]
    y = df['is_successful']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            metrics["ROC-AUC"] = float(roc_auc_score(y_test, y_prob))

        # Save model
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl"))

        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            try:
                cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_names = cat_encoder.get_feature_names_out(cat_cols)
                feature_names = safe_num_cols + list(cat_names)
                importances = model.feature_importances_
                fi_dict = dict(zip(feature_names, importances))
                feature_importance = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)[:10])
            except:
                pass

        results[name] = {"metrics": metrics, "feature_importance": feature_importance}

    return results

def get_model(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_').lower()}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None
