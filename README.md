# Renewable Energy Dashboard (Wind + Solar, combined)

This merges your two existing projects into **one Flask app with one dashboard**:

- `wind_app/` — your original Wind Energy Forecasting System, unchanged
  (data, models, ML pipeline in `wind_app/src/`).
- `solar_app/` — your original Solar Energy System, unchanged
  (`solar_app/modules/` has the current-conditions model and the
  date/time forecasting model).
- `app.py` — **new** top-level Flask app. It imports the functions from
  both projects (no calculation logic was rewritten) and renders a single
  page with a Wind tab and a Solar tab.
- `templates/dashboard.html`, `static/` — the new unified UI.

## What the dashboard shows

**Top strip:** live KPI cards — current wind power/energy, current solar
power/energy, and combined output.

**Wind tab**
- Year/month/day lookup form (historical years use real dataset averages,
  future years use the best-performing ML model, selected by RMSE from
  Linear Regression, Random Forest, and Gradient Boosting).
- Result card (wind speed, power, energy).
- A chart of average wind speed by year.
- Model performance table (MAE / RMSE / R²).

**Solar tab**
- Current-conditions form (irradiance, temperature, humidity, wind,
  cloud cover, panel area, efficiency, operating hours) → predicted
  power/energy, using your trained regression model plus the physics
  equation for comparison.
- Date/time-only forecast form → predicted temperature, irradiance, and
  expected output (kW/kWp), learned purely from historical seasonal/
  diurnal patterns (no manual reading required), exactly like your
  original `/forecast` route.
- A 24-hour forecast chart (irradiance + expected output).
- Model performance table.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000/**.

Both ML pipelines (wind model comparison, solar current-conditions model,
solar forecast model) train once at startup and are reused for every
request, same as in the original two apps.

## Deploying to Vercel

The root `vercel.json` routes every request to the Flask app, so the wind
and solar dashboard share one deployment URL. The `api/index.py` file is
the Vercel serverless entry point. Only the dashboard's required templates,
static assets, model files, and datasets are included with the function.
Unused XGBoost and Matplotlib dependencies, model artifacts, and generated
Python cache files are excluded to keep the serverless function within
Vercel's bundle-size limit. Wind model comparison uses scikit-learn's
Gradient Boosting model instead.

In Vercel, import this repository and set **Root Directory** to `combined`
(the folder containing this README and `vercel.json`). Vercel installs the
dependencies from `requirements.txt` and deploys the app. After deployment,
open the assigned URL to use the combined dashboard.

## Notes

- The original `wind_app/README.md` and `solar_app/README.md` are kept
  for reference on the underlying equations and module layout.
- The solar system's trained models/historical archive persist to
  `solar_app/models/` and `solar_app/data/` (via Joblib/CSV), same as
  before.
- The wind system's trained models persist to `wind_app/models/`.
