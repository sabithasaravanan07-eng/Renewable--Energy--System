from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import MODEL_DIR
from src.feature_engineering import build_feature_matrix


def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X, y)
    return model


def train_gradient_boosting(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": rmse,
        "R2": r2_score(y_test, predictions),
    }


def train_and_compare_models(df: pd.DataFrame, target_column: str = "wind_speed") -> tuple[dict, dict, str]:
    X, y = build_feature_matrix(df, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "Gradient Boosting": train_gradient_boosting(X_train, y_train),
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)

    best_model_name = min(results, key=lambda name: results[name]["RMSE"])
    return results, models, best_model_name


def save_model(model, name: str) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
    with open(path, "wb") as file:
        pickle.dump(model, file)
    return str(path)
