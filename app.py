"""Unified Renewable Energy Dashboard.

Combines the existing Wind Energy Forecasting System (wind_app/) and the
existing Solar Energy System (solar_app/) into a single Flask application
with one shared dashboard page. Neither project's calculation logic is
rewritten -- this file only imports and orchestrates the two existing
codebases and renders them side by side.

Run with:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000/
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WIND_DIR = os.path.join(BASE_DIR, "wind_app")
SOLAR_DIR = os.path.join(BASE_DIR, "solar_app")

# Both sub-projects use plain (non-package-relative) imports internally
# (e.g. "from config import ...", "from modules.x import y"), so each
# project root needs to be importable on sys.path -- exactly the way each
# app worked on its own.
sys.path.insert(0, WIND_DIR)
sys.path.insert(0, SOLAR_DIR)

# ---------------------------------------------------------------------------
# Wind Energy System imports (from wind_app/)
# ---------------------------------------------------------------------------
import config as wind_config  # noqa: E402
from src.data_preprocessing import load_dataset  # noqa: E402
from src.train_models import train_and_compare_models  # noqa: E402
from src.forecasting import forecast_year  # noqa: E402
from src.wind_power import calculate_wind_power, calculate_energy_output  # noqa: E402
from src.utils import get_year_range  # noqa: E402

# ---------------------------------------------------------------------------
# Solar Energy System imports (from solar_app/)
# ---------------------------------------------------------------------------
from modules.input_handler import parse_and_validate  # noqa: E402
from modules.data_processing import clean_input, quality_summary  # noqa: E402
from modules.solar_prediction import predict_solar_conditions  # noqa: E402
from modules.weather_prediction import predict_weather  # noqa: E402
from modules.power_calculation import calculate_power  # noqa: E402
from modules.energy_estimation import estimate_energy  # noqa: E402
from modules.result_analysis import analyze_result  # noqa: E402
from modules.prediction import predict_power, train_and_compare  # noqa: E402
from modules.forecasting import (  # noqa: E402
    get_forecast_model,
    forecast_for_datetime,
    generate_future_forecast,
)

# sys.path was only needed to satisfy the two projects' internal imports;
# remove the entries again so nothing from wind_app/solar_app leaks into
# unrelated third-party imports later in the process lifetime.
sys.path.remove(WIND_DIR)
sys.path.remove(SOLAR_DIR)

app = Flask(__name__)
app.config["SECRET_KEY"] = "local-development-key"

# ---------------------------------------------------------------------------
# Train / load both systems ONCE at startup (mirrors each original app.py)
# ---------------------------------------------------------------------------
WIND_DATASET = load_dataset()
WIND_MODEL_RESULTS, WIND_MODELS, WIND_BEST_MODEL_NAME = train_and_compare_models(WIND_DATASET)
WIND_BEST_MODEL = WIND_MODELS[WIND_BEST_MODEL_NAME]
WIND_MIN_YEAR, WIND_MAX_YEAR = get_year_range(WIND_DATASET)

SOLAR_POWER_ARTIFACT = train_and_compare()          # current-conditions model
SOLAR_FORECAST_ARTIFACT = get_forecast_model()       # date/time forecast model

DEFAULT_SOLAR_INPUT = {
    "irradiance_w_m2": "850",
    "temperature_c": "25",
    "humidity_percent": "55",
    "wind_kmh": "8",
    "cloud_cover_percent": "20",
    "panel_area": "20",
    "efficiency": "0.20",
    "operating_hours": "6",
}


# ---------------------------------------------------------------------------
# Wind helpers (adapted from wind_app/app.py, restored to the richer
# structure of the original wind_app/templates/index.html: full datetime
# picker, optional weather-condition overrides, full model comparison
# table, and clearly labelled estimation cards.)
# ---------------------------------------------------------------------------
WIND_CONDITION_FIELDS = ["wind_direction", "temperature", "pressure", "humidity"]

# Rough estimation-confidence banding, purely descriptive, so the dashboard
# can label how much to trust a given number instead of just showing a bare
# figure (this is the "clear estimation" piece the dashboard was missing).
def estimation_confidence(status: str) -> dict:
    if status == "Historical":
        return {"label": "High confidence", "note": "Backed by measured historical records for this period.", "level": "high"}
    if status == "Current Year":
        return {"label": "Medium-high confidence", "note": "Based on the latest recorded data for the current year.", "level": "medium-high"}
    return {"label": "Model estimate", "note": "Projected by the best-performing ML model; treat as an estimate, not a measurement.", "level": "estimate"}


def compute_wind_result(datetime_value=None, manual_inputs=None):
    manual_inputs = manual_inputs or {}

    if datetime_value:
        try:
            selected_dt = datetime.fromisoformat(str(datetime_value))
        except ValueError:
            selected_dt = datetime(wind_config.DEFAULT_YEAR, 1, 1)
    else:
        selected_dt = datetime(wind_config.DEFAULT_YEAR, 1, 1)

    year_value, month_value, day_value = selected_dt.year, selected_dt.month, selected_dt.day

    historical_end_year = 2025
    current_year = 2026
    if year_value <= historical_end_year:
        status = "Historical"
    elif year_value == current_year:
        status = "Current Year"
    else:
        status = "Future Forecast"

    manual_speed = manual_inputs.get("wind_speed")
    is_forecast_row = False
    if manual_speed not in (None, ""):
        speed = float(manual_speed)
        detail = "Actual value provided by the user for this date."
    elif status in {"Historical", "Current Year"}:
        subset = WIND_DATASET[(WIND_DATASET["year"] == year_value) & (WIND_DATASET["month"] == month_value)]
        if subset.empty:
            subset = WIND_DATASET[WIND_DATASET["year"] == year_value]
        if subset.empty:
            subset = WIND_DATASET
        speed = float(subset["wind_speed"].mean())
        detail = "Actual wind data for the requested date." if status == "Historical" else "Actual current-year wind data for the selected period."
    else:
        forecast = forecast_year(WIND_DATASET, year_value, WIND_BEST_MODEL, month_value=month_value, day_value=day_value)
        speed = float(forecast["predicted_wind_speed"])
        detail = "Forecasted value generated from the best-performing machine learning model."
        is_forecast_row = True

    power = float(calculate_wind_power(speed))
    energy = float(calculate_energy_output(power, hours=wind_config.HOURS_PER_MONTH))

    # Optional weather-condition context supplied by the user (kept for
    # display/context, matching the original app's manual-input fields).
    conditions = {}
    for field in WIND_CONDITION_FIELDS:
        raw = manual_inputs.get(field)
        if raw not in (None, ""):
            try:
                conditions[field] = float(raw)
            except (TypeError, ValueError):
                pass

    all_models = {
        name: {
            "mae": round(values["MAE"], 4),
            "rmse": round(values["RMSE"], 4),
            "r2": round(values["R2"], 4),
            "is_best": name == WIND_BEST_MODEL_NAME,
        }
        for name, values in WIND_MODEL_RESULTS.items()
    }

    # Simple visual scaling for progress-bar style metric cards (0-100).
    speed_pct = min(speed / 25 * 100, 100)
    power_pct = min(power / 15000 * 100, 100)
    energy_pct = min(energy / 10000 * 100, 100)

    return {
        "year": year_value,
        "month": month_value,
        "day": day_value,
        "date": selected_dt.strftime("%Y-%m-%d"),
        "time": selected_dt.strftime("%H:%M"),
        "status": status,
        "is_forecast": is_forecast_row,
        "wind_speed": round(speed, 3),
        "wind_power_w": round(power, 2),
        "energy_kwh": round(energy, 2),
        "detail": detail,
        "confidence": estimation_confidence(status),
        "conditions": conditions,
        "model_name": WIND_BEST_MODEL_NAME,
        "mae": round(WIND_MODEL_RESULTS[WIND_BEST_MODEL_NAME]["MAE"], 4),
        "rmse": round(WIND_MODEL_RESULTS[WIND_BEST_MODEL_NAME]["RMSE"], 4),
        "r2": round(WIND_MODEL_RESULTS[WIND_BEST_MODEL_NAME]["R2"], 4),
        "all_models": all_models,
        "min_year": WIND_MIN_YEAR,
        "max_year": WIND_MAX_YEAR,
        "speed_pct": round(speed_pct, 1),
        "power_pct": round(power_pct, 1),
        "energy_pct": round(energy_pct, 1),
    }


def wind_history_chart_data(selected_year=None):
    yearly = WIND_DATASET.groupby("year")["wind_speed"].mean().tail(15)
    return {
        "labels": [int(y) for y in yearly.index],
        "values": [round(float(v), 2) for v in yearly.values],
        "selected_year": selected_year,
    }


def wind_model_comparison_chart_data():
    names = list(WIND_MODEL_RESULTS.keys())
    return {
        "labels": names,
        "rmse": [round(WIND_MODEL_RESULTS[n]["RMSE"], 4) for n in names],
        "mae": [round(WIND_MODEL_RESULTS[n]["MAE"], 4) for n in names],
        "r2": [round(WIND_MODEL_RESULTS[n]["R2"], 4) for n in names],
        "best": WIND_BEST_MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Solar helpers (adapted from solar_app/app.py)
# ---------------------------------------------------------------------------
def compute_solar_current(form):
    values, errors = parse_and_validate(form)
    if errors:
        return None, errors
    cleaned, processing = clean_input(values)
    record = cleaned.iloc[0].to_dict()
    record.setdefault("ambient_temperature", 25.0)

    solar = predict_solar_conditions(record)
    weather = predict_weather(record)
    predicted_power = predict_power(record, SOLAR_POWER_ARTIFACT)
    physical_power = calculate_power(record["panel_area"], solar["effective_irradiance_w_m2"], record["efficiency"])
    energy = estimate_energy(predicted_power, record["operating_hours"])
    analysis = analyze_result(record, predicted_power, energy, SOLAR_POWER_ARTIFACT["metrics"], solar)

    result = {
        "input": record,
        "processing": {**processing, **quality_summary(cleaned)},
        "model_name": SOLAR_POWER_ARTIFACT["model_name"],
        "metrics": SOLAR_POWER_ARTIFACT["metrics"],
        "predicted_power_w": round(predicted_power, 2),
        "physical_power_w": round(physical_power, 2),
        "solar": solar,
        "weather": weather,
        "energy": {k: round(v, 2) for k, v in energy.items()},
        "analysis": analysis,
    }
    return result, []


def default_solar_forecast():
    target_dt = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=18)
    return forecast_for_datetime(SOLAR_FORECAST_ARTIFACT, target_dt)


def solar_forecast_chart_data():
    series = generate_future_forecast(SOLAR_FORECAST_ARTIFACT, "24_hours")
    return {
        "labels": [p["timestamp"][11:16] for p in series["points"]],
        "irradiance": [p["irradiance_w_m2"] for p in series["points"]],
        "output": [p["predicted_output_kw_per_kwp"] for p in series["points"]],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
DEFAULT_WIND_DATETIME = f"{wind_config.DEFAULT_YEAR}-01-01T00:00"


@app.route("/", methods=["GET"])
def dashboard():
    wind_result = compute_wind_result(DEFAULT_WIND_DATETIME)
    solar_current, _ = compute_solar_current(DEFAULT_SOLAR_INPUT)
    solar_forecast = default_solar_forecast()
    return render_template(
        "dashboard.html",
        wind_result=wind_result,
        wind_chart=wind_history_chart_data(wind_result["year"]),
        wind_model_chart=wind_model_comparison_chart_data(),
        wind_form={},
        wind_errors=[],
        solar_current=solar_current,
        solar_forecast=solar_forecast,
        solar_chart=solar_forecast_chart_data(),
        solar_form=DEFAULT_SOLAR_INPUT,
        solar_errors=[],
        active_panel=request.args.get("panel", "wind"),
    )


@app.route("/wind/query", methods=["POST"])
def wind_query():
    datetime_value = request.form.get("selected_datetime") or DEFAULT_WIND_DATETIME
    manual_inputs = {
        "wind_speed": request.form.get("wind_speed"),
        "wind_direction": request.form.get("wind_direction"),
        "temperature": request.form.get("temperature"),
        "pressure": request.form.get("pressure"),
        "humidity": request.form.get("humidity"),
    }
    wind_result = compute_wind_result(datetime_value, manual_inputs=manual_inputs)

    solar_current, _ = compute_solar_current(DEFAULT_SOLAR_INPUT)
    solar_forecast = default_solar_forecast()
    return render_template(
        "dashboard.html",
        wind_result=wind_result,
        wind_chart=wind_history_chart_data(wind_result["year"]),
        wind_model_chart=wind_model_comparison_chart_data(),
        wind_form=request.form,
        wind_errors=[],
        solar_current=solar_current,
        solar_forecast=solar_forecast,
        solar_chart=solar_forecast_chart_data(),
        solar_form=DEFAULT_SOLAR_INPUT,
        solar_errors=[],
        active_panel="wind",
    )


@app.route("/solar/current", methods=["POST"])
def solar_current_query():
    solar_current, errors = compute_solar_current(request.form)
    wind_result = compute_wind_result(DEFAULT_WIND_DATETIME)
    solar_forecast = default_solar_forecast()
    return render_template(
        "dashboard.html",
        wind_result=wind_result,
        wind_chart=wind_history_chart_data(wind_result["year"]),
        wind_model_chart=wind_model_comparison_chart_data(),
        wind_form={},
        wind_errors=[],
        solar_current=solar_current,
        solar_forecast=solar_forecast,
        solar_chart=solar_forecast_chart_data(),
        solar_form=request.form,
        solar_errors=errors,
        active_panel="solar",
    )


@app.route("/solar/forecast", methods=["POST"])
def solar_forecast_query():
    error = None
    solar_forecast = None
    target_str = request.form.get("forecast_datetime")
    try:
        if not target_str:
            raise ValueError("Please choose a future date and time.")
        target_dt = datetime.fromisoformat(target_str)
        solar_forecast = forecast_for_datetime(SOLAR_FORECAST_ARTIFACT, target_dt)
    except Exception as exc:
        error = str(exc)
        solar_forecast = default_solar_forecast()

    wind_result = compute_wind_result(DEFAULT_WIND_DATETIME)
    solar_current, _ = compute_solar_current(DEFAULT_SOLAR_INPUT)
    return render_template(
        "dashboard.html",
        wind_result=wind_result,
        wind_chart=wind_history_chart_data(wind_result["year"]),
        wind_model_chart=wind_model_comparison_chart_data(),
        wind_form={},
        wind_errors=[],
        solar_current=solar_current,
        solar_forecast=solar_forecast,
        solar_forecast_error=error,
        solar_chart=solar_forecast_chart_data(),
        solar_form=DEFAULT_SOLAR_INPUT,
        solar_errors=[],
        active_panel="solar",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
