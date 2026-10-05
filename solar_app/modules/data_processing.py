"""Cleaning, duplicate/missing checks, and feature summary utilities.

The feature list here is intentionally isolated so additional parameters
can be added without touching the rest of the pipeline (see README).
"""
import pandas as pd

FEATURES = [
    "irradiance_w_m2",
    "temperature_c",
    "humidity_percent",
    "wind_kmh",
    "cloud_cover_percent",
    "panel_area",
    "efficiency",
    "operating_hours",
]

DEFAULTS = {
    "ambient_temperature": 25.0,
}


def clean_input(values: dict):
    """Turn a single validated values dict into a one-row DataFrame,
    applying default fills and reporting what was done.
    """
    df = pd.DataFrame([values])

    missing_before = int(df.isna().sum().sum())
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    duplicates_found = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_after_coerce = int(df.isna().sum().sum())
    df = df.fillna(df.median(numeric_only=True))

    processing = {
        "rows_in": 1,
        "rows_out": len(df),
        "missing_values_filled": missing_before + missing_after_coerce,
        "duplicate_rows_removed": duplicates_found,
    }
    return df, processing


def quality_summary(cleaned: pd.DataFrame) -> dict:
    numeric = cleaned.select_dtypes("number")
    return {
        "feature_count": len([c for c in FEATURES if c in cleaned.columns]),
        "value_ranges": {
            col: {"min": round(float(numeric[col].min()), 3), "max": round(float(numeric[col].max()), 3)}
            for col in numeric.columns
        },
    }
