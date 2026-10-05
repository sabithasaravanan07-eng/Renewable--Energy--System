from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        valid_dates = df["date"].notna()
        df.loc[valid_dates, "day_of_year"] = df.loc[valid_dates, "date"].dt.dayofyear
        df.loc[valid_dates, "quarter"] = df.loc[valid_dates, "date"].dt.quarter
        df.loc[valid_dates, "is_winter"] = df.loc[valid_dates, "date"].dt.month.isin([12, 1, 2]).astype(int)
        df.loc[valid_dates, "is_summer"] = df.loc[valid_dates, "date"].dt.month.isin([6, 7, 8]).astype(int)
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    return df


def build_feature_matrix(df: pd.DataFrame, target_column: str = "wind_speed") -> tuple[pd.DataFrame, pd.Series]:
    feature_df = add_time_features(df).copy()

    feature_columns = [
        col for col in [
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
        if col in feature_df.columns
    ]

    X = feature_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(feature_df[target_column], errors="coerce")
    y = y.fillna(y.median())
    return X, y
