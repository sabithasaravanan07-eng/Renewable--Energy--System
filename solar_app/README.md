# Solar Energy System

A modular, college-level Flask application for estimating solar power and energy from user-provided solar, environmental, and system parameters. It validates and cleans inputs, compares regression models, persists the selected model with Joblib, calculates physics-based power and energy, and presents an interactive dashboard. It also forecasts a future date/time purely from learned historical (past) data patterns.

## Features

- Flask input and result dashboard
- Validation for numeric ranges and date/time
- Missing-value, duplicate, invalid-value, and feature-processing utilities
- Linear Regression, Random Forest Regression, and Gradient Boosting comparison
- MAE, MSE, RMSE, and R2 evaluation metrics
- Automatic best-model selection by R2 score
- Joblib model persistence in `models/solar_power_model.joblib`
- Solar-condition adjustment, weather insight, power and energy calculations
- Chart.js result visualization and JSON report in `reports/latest_report.json`
- **Date/time-only forecasting**: predicts temperature, irradiance, and expected output for any future date and time using patterns learned from historical (past) data — no live reading is typed in
- Extensible feature list in `modules/data_processing.py`

## Installation

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/.

## VS Code execution

1. Open the `solar_energy_system` folder in VS Code.
2. Select the `.venv` interpreter with `Python: Select Interpreter`.
3. Open the integrated terminal and activate the environment.
4. Run `python app.py`.
5. Open the local URL printed by Flask.

## Workflow

Two independent flows:

**Current-conditions estimate** (`/input`):
`Input -> Validation -> Cleaning -> Feature preparation -> Model comparison -> Solar/weather insights -> Power -> Energy -> Analysis -> Dashboard/report`

**Forecast** (`/forecast`) — redesigned to depend only on date and time:
`Historical (past) data (persisted, seasonal + diurnal pattern) -> Feature engineering from month/day-of-year/hour -> Model comparison -> Pick a future date & time -> Predicted temperature, irradiance, expected output`

The forecast page never asks for a live temperature/irradiance reading. You either:
1. Pick a **future date & time** and get a single-point forecast, or
2. Pick a **future period** (24 hours / 3 days / 7 days) and get an hourly outlook, or
3. Optionally paste your own historical CSV (timestamp, temperature_c, irradiance_w_m2, ...) to retrain the pattern instead of using the built-in synthetic archive.

The historical archive is generated once and persisted to `data/historical_solar_data.csv` (one year of hourly data ending "yesterday"), so it behaves like a stable past-data archive rather than fresh random numbers on every run. Replace `_generate_synthetic_history()` in `modules/forecasting.py` with a real data loader when field-measured historical data is available.

The current-conditions model trains the candidate models from a reproducible physics-informed local training dataset. The best model and its metrics are then stored with Joblib and reused on later requests. This dataset is a demonstration baseline, not a claim about field accuracy; replace `_training_data()` in `modules/prediction.py` with measured historical data for a production model.

## Equations

Power uses `P = A x G x eta x r`, where `A` is panel area in m2, `G` is effective irradiance in W/m2, `eta` is conversion efficiency, and `r` is a configurable performance ratio currently set to 0.90 in the calculation function. Temperature adjusts `G` around a 25 C reference using a bounded temperature coefficient: warmer conditions reduce panel output and cooler conditions slightly improve it. Temperature alone cannot produce energy without irradiance.

Energy uses `E = P x t`. Power is reported in watts, time in hours, and energy in watt-hours and kilowatt-hours.

Forecast expected output is expressed as **kW per kWp installed** (`irradiance / 1000 W/m2 * derate factor`), since a forecast has no specific panel area/efficiency attached to it — multiply by your installed capacity in kWp to get expected kW.

## Module guide

- `input_handler.py`: parses and validates form values for the current-conditions estimate.
- `data_processing.py`: cleaning, duplicate/missing checks, and feature definitions.
- `prediction.py`: current-conditions power model — training, comparison, persistence, inference.
- `forecasting.py`: date/time-only forecasting — historical data generation/parsing, feature engineering (month/day-of-year/hour), model comparison, and future-date/period prediction.
- `weather_prediction.py`: transparent environmental indicators.
- `solar_prediction.py`: effective irradiance adjustments.
- `power_calculation.py`: physical power equation.
- `energy_estimation.py`: energy equation and unit conversion.
- `result_analysis.py`: observations and chart payload.
- `reporting.py`: latest JSON report export.
- `app.py`: Flask orchestration and routes.

## Testing

Compile the source modules:

```powershell
python -m py_compile app.py modules\*.py
```

For the current-conditions estimate, submit the `/input` form with values such as irradiance 850 W/m2, temperature 25 C, humidity 55%, wind 8 km/h, cloud cover 20%, area 20 m2, efficiency 0.20, and 6 hours.

For the forecast, go to `/forecast` and pick any future date/time — no readings needed. Try noon in a summer month vs. midnight in a winter month to see the seasonal/diurnal pattern in action.

For production use, swap the synthetic historical archive for measured, timestamped historical observations (via the CSV paste box, or by editing `demo_historical_data()`). The feature list and model candidate registry are intentionally isolated so additional parameters and estimators can be added without redesigning the dashboard.
