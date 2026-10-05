# Wind Energy Forecasting and Prediction System

This project provides a complete AI/ML-based wind energy forecasting solution for both historical and future years. The application lets users enter a year and determine whether the data is historical or forecasted, then calculates wind power and energy output using the physical wind energy equation.

## Project objective

The system handles:

1. Historical years with actual wind and weather data
2. Future years without actual data, which require forecasting
3. Model comparison and best-model selection
4. Dashboard visualization and year-based analysis

## Core workflow

Historical dataset -> Data preprocessing -> Feature engineering -> Mathematical wind power calculation -> Model training -> Evaluation -> User year input -> Historical or forecast branch -> Wind power and energy calculation -> Dashboard output

## Requirements

- Python 3.10+
- Flask
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Plotly
- Joblib

## Folder structure

- `data/raw/`: Raw historical dataset files
- `data/processed/`: Cleaned and processed data files
- `models/`: Saved trained model objects
- `notebooks/`: Experiment notebooks
- `src/`: Core data science and forecasting logic
- `templates/`: Flask HTML templates
- `static/`: CSS and JavaScript assets
- `app.py`: Flask application entry point
- `requirements.txt`: Python dependency list
- `config.py`: Global project settings
- `README.md`: Documentation

## Historical year logic

If the requested year is available within the dataset, the system uses actual wind and weather measurements, computes wind power with the equation:

P = 1/2 * rho * A * v^3 * Cp

and converts to energy in kWh.

## Future year logic

If the requested year is outside the dataset range, the system trains on historical data, predicts wind speed for the future year, then applies the same physical equation to compute forecasted power and energy.

## ML models

The project compares these regressors:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

It evaluates each model with:

- MAE
- RMSE
- R²

The best model is selected by the lowest RMSE.

## Running the app

1. Install dependencies:
   `pip install -r requirements.txt`
2. Start the Flask app:
   `python app.py`
3. Open the browser at:
   `http://localhost:5000`

## Data flow

- Raw CSV data enters `src/data_preprocessing.py`
- Feature engineering is prepared in `src/feature_engineering.py`
- Mathematical calculation occurs in `src/wind_power.py`
- Model training and selection occur in `src/train_models.py`
- Forecasting logic is handled in `src/forecasting.py`
- Flask UI consumes the results from `app.py`

## Notes

The preprocessing layer is designed to be dataset-configurable, so it does not depend on a fixed historical year range. The app automatically detects the dataset's minimum and maximum available years.

## Future improvements

- Add a real database (SQLite/PostgreSQL)
- Include Plotly charts for actual vs predicted data
- Add API endpoints for external system integration
- Add feature importance and error analysis
- Add model drift monitoring and retraining workflow
