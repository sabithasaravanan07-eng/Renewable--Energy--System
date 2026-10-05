"""Trains and compares regression models on a reproducible,
physics-informed synthetic training set, persists the best model with
Joblib, and reuses it on later requests instead of retraining every time.

This model estimates instantaneous power from CURRENT snapshot
conditions (irradiance, temperature, etc.) submitted on the /input page.
It is unrelated to the date/time-only forecasting feature in
modules/forecasting.py.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "solar_power_model.joblib")

FEATURE_NAMES = [
    "irradiance_w_m2",
    "temperature_c",
    "humidity_percent",
    "wind_kmh",
    "cloud_cover_percent",
    "panel_area",
    "efficiency",
]

PERFORMANCE_RATIO = 0.90


def _training_data(n=1200, seed=7) -> pd.DataFrame:
    """Reproducible, physics-informed synthetic dataset. This is a
    demonstration baseline only (see README) -- swap in measured
    historical observations for a production model.
    """
    rng = np.random.default_rng(seed)
    irradiance = rng.uniform(0, 1200, n)
    temperature = rng.uniform(-5, 45, n)
    humidity = rng.uniform(10, 100, n)
    wind = rng.uniform(0, 60, n)
    cloud = rng.uniform(0, 100, n)
    panel_area = rng.uniform(1, 60, n)
    efficiency = rng.uniform(0.10, 0.23, n)

    temp_coeff = -0.0045
    effective_irradiance = irradiance * (1 - cloud / 140.0)
    temp_factor = 1 + temp_coeff * (temperature - 25.0)
    power = panel_area * effective_irradiance * efficiency * PERFORMANCE_RATIO * temp_factor
    power = np.clip(power, 0, None)
    power += rng.normal(0, power.std() * 0.03 + 1e-6, n)
    power = np.clip(power, 0, None)

    df = pd.DataFrame(
        {
            "irradiance_w_m2": irradiance,
            "temperature_c": temperature,
            "humidity_percent": humidity,
            "wind_kmh": wind,
            "cloud_cover_percent": cloud,
            "panel_area": panel_area,
            "efficiency": efficiency,
            "power_w": power,
        }
    )
    return df


def _candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(n_estimators=150, random_state=7),
        "Gradient Boosting Regression": GradientBoostingRegressor(random_state=7),
    }


def _train_fresh():
    data = _training_data()
    X = data[FEATURE_NAMES]
    y = data["power_w"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    best_name, best_model, best_metrics, best_r2 = None, None, None, -np.inf
    all_metrics = {}
    for name, model in _candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)
        metrics = {"mae": round(mae, 4), "mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
        all_metrics[name] = metrics
        if r2 > best_r2:
            best_name, best_model, best_metrics, best_r2 = name, model, metrics, r2

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "metrics": best_metrics,
        "all_metrics": all_metrics,
        "feature_names": FEATURE_NAMES,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def train_and_compare():
    """Loads the persisted model if present, otherwise trains, compares,
    and persists a fresh one. This satisfies the "train once, reuse on
    later requests" behaviour described in the README.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return _train_fresh()


def predict_power(record: dict, artifact: dict) -> float:
    row = pd.DataFrame([{name: record[name] for name in artifact["feature_names"]}])
    prediction = float(artifact["model"].predict(row)[0])
    return max(prediction, 0.0)
