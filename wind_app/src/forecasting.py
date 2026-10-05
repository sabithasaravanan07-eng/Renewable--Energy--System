from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import add_time_features


def prepare_future_features(df: pd.DataFrame, target_year: int, model=None, target_month: int = 1, target_day: int = 1) -> pd.DataFrame:
    """Generate one row of features for a target future date aligned to the model input schema."""
    target_date = pd.Timestamp(target_year, target_month, target_day)
    future_frame = pd.DataFrame({"date": [target_date]})
    future_frame["year"] = target_year
    future_frame["month"] = int(target_month)
    future_frame = add_time_features(future_frame)

    for col in ["wind_direction", "temperature", "pressure", "humidity"]:
        if col in df.columns:
            future_frame[col] = float(df[col].mean())
        else:
            future_frame[col] = 0.0

    feature_columns = [
        "year",
        "month",
        "day_of_year",
        "quarter",
        "is_winter",
        "is_summer",
        "wind_direction",
        "temperature",
        "pressure",
        "humidity",
    ]

    if model is not None and hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    for col in feature_columns:
        if col not in future_frame.columns:
            future_frame[col] = 0.0

    return future_frame[feature_columns]


def forecast_year(df: pd.DataFrame, year_value: int, model, target_column: str = "wind_speed", month_value: int = 1, day_value: int = 1) -> dict:
    feature_df = add_time_features(df.copy())
    requested_month = int(month_value)
    requested_day = int(day_value)

    if year_value in feature_df["year"].unique():
        subset = feature_df[(feature_df["year"] == year_value) & (feature_df["month"] == requested_month)]
        if not subset.empty:
            actual_speed = float(subset[target_column].mean())
            return {
                "status": "Historical",
                "year": year_value,
                "actual_wind_speed": actual_speed,
                "predicted_wind_speed": actual_speed,
                "is_forecast": False,
            }

    future_features = prepare_future_features(df, year_value, model=model, target_month=requested_month, target_day=requested_day)
    prediction = float(model.predict(future_features)[0])
    return {
        "status": "Future Forecast",
        "year": year_value,
        "actual_wind_speed": None,
        "predicted_wind_speed": prediction,
        "is_forecast": True,
    }
