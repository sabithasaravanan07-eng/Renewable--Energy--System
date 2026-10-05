"""Date/time-driven solar forecasting.

Design goal (per project requirement): forecasting must NOT depend on a
user manually typing in a temperature/irradiance reading, the way a
real-time sensor snapshot would. Instead, the forecast for any future
date and time is produced from PAST (historical) data patterns keyed
only by month / day-of-year / hour. Two inputs drive every forecast:
the target date and the target time -- nothing else.

Pipeline:
  1. demo_historical_data()   -> builds (once) and persists a full year
                                  of past hourly observations to
                                  data/historical_solar_data.csv, so the
                                  "history" a user sees is stable across
                                  runs, exactly like a real historical
                                  archive would be.
  2. parse_historical_csv()   -> lets an advanced user supply their own
                                  real historical CSV instead of the
                                  synthetic archive.
  3. build_forecast_model()   -> learns the seasonal (month/day-of-year)
                                  and diurnal (hour-of-day) pattern of
                                  temperature and irradiance from that
                                  historical data, comparing a few
                                  regressors and keeping the best.
  4. forecast_for_datetime()  -> given ONLY a future date+time, predicts
                                  temperature, irradiance and expected
                                  solar output from the learned pattern.
  5. generate_future_forecast() -> same idea, but produces an hourly
                                  series over a future period (24h / 3d /
                                  7d) starting from now.
"""
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "historical_solar_data.csv")
FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.joblib")

TARGET_COLUMNS = ["temperature_c", "irradiance_w_m2"]
FEATURE_COLUMNS = ["month_sin", "month_cos", "doy_sin", "doy_cos", "hour_sin", "hour_cos"]

# Reference "installed capacity" used only to express solar output in kW
# for the forecast view (irradiance -> expected generation per kW installed).
REFERENCE_DERATE = 0.80  # inverter + system losses
STC_IRRADIANCE = 1000.0  # W/m2, standard test condition reference

PERIOD_HOURS = {
    "24_hours": (24, 1),
    "3_days": (72, 3),
    "7_days": (168, 6),
}


# ---------------------------------------------------------------------------
# Historical ("past") data
# ---------------------------------------------------------------------------

def _generate_synthetic_history(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Synthesises one year of PAST hourly observations, ending yesterday,
    with realistic seasonal (annual) and diurnal (daily) patterns. This
    stands in for a measured historical archive -- swap this function for
    a real data loader when field data is available.
    """
    rng = np.random.default_rng(seed)
    end = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start = end - timedelta(days=days)
    timestamps = pd.date_range(start=start, end=end, freq="h")

    day_of_year = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()

    # Seasonal component: peak around mid-year, trough around year start.
    seasonal = np.cos(2 * np.pi * (day_of_year - 172) / 365.25)

    # Diurnal irradiance: daylight roughly 06:00-18:00, peak at solar noon.
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0, None)
    seasonal_irr_factor = 0.75 + 0.25 * seasonal  # tropical: mild seasonal swing
    clear_sky_irradiance = daylight * seasonal_irr_factor * 950.0

    # Cloud cover: baseline random cloudiness, heavier in "monsoon" months.
    monsoon_boost = np.clip(-np.cos(2 * np.pi * (day_of_year - 260) / 365.25), 0, None) * 35
    cloud_cover = np.clip(rng.normal(30 + monsoon_boost, 15), 0, 100)

    irradiance = clip_irradiance = clear_sky_irradiance * (1 - cloud_cover / 140.0)
    irradiance = np.clip(irradiance + rng.normal(0, 25, len(timestamps)), 0, 1200)
    irradiance = np.where(daylight <= 0, 0.0, irradiance)

    # Temperature: annual + diurnal sinusoids plus noise.
    annual_temp = 27 + 5 * seasonal
    diurnal_temp = 5 * np.sin(np.pi * (hour - 6) / 12.0)
    temperature = annual_temp + diurnal_temp + rng.normal(0, 1.3, len(timestamps))

    humidity = np.clip(70 - 0.15 * (temperature - 27) + 0.25 * cloud_cover + rng.normal(0, 5, len(timestamps)), 20, 100)
    wind = np.clip(rng.normal(10 + 0.05 * cloud_cover, 4), 0, 45)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": temperature,
            "irradiance_w_m2": irradiance,
            "cloud_cover_percent": cloud_cover,
            "humidity_percent": humidity,
            "wind_kmh": wind,
        }
    )
    return df


def demo_historical_data(days: int = 365) -> pd.DataFrame:
    """Returns the persisted historical archive, generating it once if it
    doesn't exist yet. Because it's persisted to CSV, it behaves like a
    stable past-data archive rather than fresh random data on every call.
    """
    if os.path.exists(HISTORICAL_DATA_PATH):
        df = pd.read_csv(HISTORICAL_DATA_PATH, parse_dates=["timestamp"])
        if len(df) > 0:
            return df
    df = _generate_synthetic_history(days=days)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(HISTORICAL_DATA_PATH, index=False)
    return df


def parse_historical_csv(csv_text: str) -> pd.DataFrame:
    """Parses a user-supplied historical CSV as an alternative to the
    synthetic archive. Required columns: timestamp, temperature_c,
    irradiance_w_m2. Optional: cloud_cover_percent, humidity_percent,
    wind_kmh.
    """
    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))
    required = {"timestamp", "temperature_c", "irradiance_w_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical CSV is missing required column(s): {', '.join(sorted(missing))}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for optional in ["cloud_cover_percent", "humidity_percent", "wind_kmh"]:
        if optional not in df.columns:
            df[optional] = np.nan
    df = df.dropna(subset=["timestamp", "temperature_c", "irradiance_w_m2"])
    if df.empty:
        raise ValueError("Historical CSV had no usable rows after parsing.")
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering (date/time ONLY -- no live readings)
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    month = ts.dt.month.to_numpy()
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy()

    out = pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        }
    )
    return out


def _features_for_datetime(target_dt: datetime) -> pd.DataFrame:
    single = pd.DataFrame({"timestamp": [target_dt]})
    return _engineer_features(single)


# ---------------------------------------------------------------------------
# Model training / comparison / persistence
# ---------------------------------------------------------------------------

def _candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=7)),
        "Gradient Boosting Regression": MultiOutputRegressor(GradientBoostingRegressor(random_state=7)),
    }


def build_forecast_model(historical_df: pd.DataFrame = None) -> dict:
    """Trains regressors that map date/time features -> (temperature,
    irradiance), i.e. purely historical seasonal + diurnal pattern
    learning. No live sensor input is used anywhere in this function.
    """
    if historical_df is None:
        historical_df = demo_historical_data()

    X = _engineer_features(historical_df)
    y = historical_df[TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    best_name, best_model, best_metrics, best_score = None, None, None, -np.inf
    all_metrics = {}
    for name, model in _candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_df = pd.DataFrame(preds, columns=TARGET_COLUMNS)
        mae_temp = mean_absolute_error(y_test["temperature_c"], preds_df["temperature_c"])
        mae_irr = mean_absolute_error(y_test["irradiance_w_m2"], preds_df["irradiance_w_m2"])
        r2_temp = r2_score(y_test["temperature_c"], preds_df["temperature_c"])
        r2_irr = r2_score(y_test["irradiance_w_m2"], preds_df["irradiance_w_m2"])
        avg_r2 = (r2_temp + r2_irr) / 2.0

        metrics = {
            "temperature_mae": round(mae_temp, 3),
            "irradiance_mae": round(mae_irr, 3),
            "temperature_r2": round(r2_temp, 4),
            "irradiance_r2": round(r2_irr, 4),
            "avg_r2": round(avg_r2, 4),
        }
        all_metrics[name] = metrics
        if avg_r2 > best_score:
            best_name, best_model, best_metrics, best_score = name, model, metrics, avg_r2

    artifact = {
        "model": best_model,
        "best_model": best_name,
        "metrics": best_metrics,
        "all_metrics": all_metrics,
        "feature_names": FEATURE_COLUMNS,
        "target_names": TARGET_COLUMNS,
        "history_rows": len(historical_df),
        "history_range": {
            "start": str(historical_df["timestamp"].min()),
            "end": str(historical_df["timestamp"].max()),
        },
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, FORECAST_MODEL_PATH)
    return artifact


def get_forecast_model() -> dict:
    """Loads the persisted forecast model if present, otherwise builds
    (and persists) one from the historical archive."""
    if os.path.exists(FORECAST_MODEL_PATH):
        try:
            return joblib.load(FORECAST_MODEL_PATH)
        except Exception:
            pass
    return build_forecast_model()


# ---------------------------------------------------------------------------
# Forecast output helpers
# ---------------------------------------------------------------------------

def _solar_output_kw(irradiance_w_m2: float) -> float:
    """Expected generation per kW of installed capacity, derived purely
    from predicted irradiance (irradiance / STC * derate)."""
    return max(irradiance_w_m2, 0.0) / STC_IRRADIANCE * REFERENCE_DERATE


def _predict_row(model_artifact: dict, features: pd.DataFrame) -> dict:
    preds = model_artifact["model"].predict(features)[0]
    temperature_c = float(preds[0])
    irradiance_w_m2 = max(float(preds[1]), 0.0)
    return {
        "temperature_c": round(temperature_c, 2),
        "irradiance_w_m2": round(irradiance_w_m2, 2),
        "predicted_output_kw_per_kwp": round(_solar_output_kw(irradiance_w_m2), 3),
    }


def forecast_for_datetime(model_artifact: dict, target_dt: datetime) -> dict:
    """The ONLY inputs are a future date and time. Temperature,
    irradiance and expected solar output are all derived from the
    learned historical pattern for that month/day/hour -- nothing is
    typed in manually.
    """
    now = datetime.now()
    if target_dt <= now:
        raise ValueError("Please choose a date and time in the future.")

    features = _features_for_datetime(target_dt)
    result = _predict_row(model_artifact, features)
    result["forecast_datetime"] = target_dt.isoformat(timespec="minutes")
    result["model_name"] = model_artifact["best_model"]
    result["metrics"] = model_artifact["metrics"]
    return result


def generate_future_forecast(model_artifact: dict, future_period: str) -> dict:
    """Produces an hourly (or multi-hourly, for longer horizons) series
    for a future period, starting from now, driven only by date/time."""
    if future_period not in PERIOD_HOURS:
        raise ValueError(f"Unknown future period '{future_period}'.")
    total_hours, step_hours = PERIOD_HOURS[future_period]

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    points = []
    for offset in range(step_hours, total_hours + 1, step_hours):
        target_dt = now + timedelta(hours=offset)
        features = _features_for_datetime(target_dt)
        row = _predict_row(model_artifact, features)
        row["timestamp"] = target_dt.isoformat(timespec="minutes")
        points.append(row)

    return {
        "future_period": future_period,
        "generated_at": now.isoformat(timespec="minutes"),
        "model_name": model_artifact["best_model"],
        "metrics": model_artifact["metrics"],
        "points": points,
    }
