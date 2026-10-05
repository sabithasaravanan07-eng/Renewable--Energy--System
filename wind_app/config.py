from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
APP_TITLE = "Wind Energy Forecasting System"

DEFAULT_DATASET = RAW_DATA_DIR / "wind_dataset.csv"
DEFAULT_TARGET_COLUMN = "wind_speed"
DEFAULT_YEAR = 2025

AIR_DENSITY = 1.225
TURBINE_SWEPT_AREA = 100.0
POWER_COEFFICIENT = 0.4
HOURS_PER_MONTH = 720
