from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from config import DEFAULT_DATASET


def normalize_column_name(name: str) -> str:
    """Standardize incoming column names to a predictable format."""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def ensure_sample_dataset(dataset_path: str | Path = DEFAULT_DATASET) -> Path:
    """Create a synthetic monthly wind dataset when no real dataset exists."""
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        return dataset_path

    dates = pd.date_range("1985-01-01", "2025-12-01", freq="MS")
    years = []
    months = []
    wind_speed = []
    wind_direction = []
    temperature = []
    pressure = []
    humidity = []

    for dt in dates:
        base_speed = 7.0 + 2.5 * (1 + ((dt.year - 2010) % 5) / 5) + 1.2 * (1 + (dt.month / 12))
        seasonal = 2.5 * (1.0 + (dt.month / 12)) * (1 + (dt.month % 2))
        noise = ((dt.day + dt.month + dt.year) % 9) / 10
        speed = max(2.0, base_speed + seasonal * 0.35 - 0.7 + noise)
        direction = (120 + dt.month * 18 + dt.year * 1.5) % 360
        temp = 18 + 12 * (1 + (dt.month / 12)) + ((dt.year - 2010) * 0.2)
        pressure_value = 1010 + 20 * (1 - (dt.month / 12)) + ((dt.year - 2010) * 0.5)
        humidity_value = 55 + 18 * (1 + (dt.month / 12)) / 2

        years.append(dt.year)
        months.append(dt.month)
        wind_speed.append(round(speed, 3))
        wind_direction.append(round(direction, 2))
        temperature.append(round(temp, 2))
        pressure.append(round(pressure_value, 2))
        humidity.append(round(humidity_value, 2))

    df = pd.DataFrame(
        {
            "date": dates,
            "year": years,
            "month": months,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,
        }
    )
    df.to_csv(dataset_path, index=False)
    return dataset_path


def get_year_range(df: pd.DataFrame) -> Tuple[int, int]:
    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    if years.empty:
        raise ValueError("No year information found in the dataset.")
    return int(years.min()), int(years.max())


def find_missing_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> List[str]:
    missing = []
    for col in required_columns:
        if col not in df.columns:
            missing.append(col)
    return missing


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if column == "date":
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        try:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        except Exception:
            pass
    return df
