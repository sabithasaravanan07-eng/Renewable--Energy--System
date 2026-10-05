from __future__ import annotations

import pandas as pd

from config import DEFAULT_DATASET
from src.utils import convert_numeric_columns, ensure_sample_dataset, normalize_columns


def load_dataset(dataset_path: str = str(DEFAULT_DATASET)) -> pd.DataFrame:
    dataset_path = ensure_sample_dataset(dataset_path)
    df = pd.read_csv(dataset_path)
    df = normalize_columns(df)
    df = convert_numeric_columns(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().all():
            if {"year", "month"}.issubset(df.columns):
                df["date"] = pd.to_datetime(
                    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
                    errors="coerce",
                )

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.index, errors="coerce")

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].replace([float("inf"), float("-inf")], pd.NA)
            df[column] = df[column].fillna(df[column].median())

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def prepare_model_data(df: pd.DataFrame, target_column: str = "wind_speed") -> pd.DataFrame:
    df = clean_dataset(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    if "month" not in df.columns and "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.month

    return df
