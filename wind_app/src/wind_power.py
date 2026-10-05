from __future__ import annotations

import numpy as np
import pandas as pd

from config import AIR_DENSITY, POWER_COEFFICIENT, TURBINE_SWEPT_AREA


def calculate_wind_power(wind_speed: float | pd.Series, rho: float = AIR_DENSITY, area: float = TURBINE_SWEPT_AREA, cp: float = POWER_COEFFICIENT) -> float | pd.Series:
    """Compute wind power using P = 0.5 * rho * A * v^3 * Cp."""
    return 0.5 * rho * area * np.power(np.asarray(wind_speed, dtype=float), 3) * cp


def calculate_energy_output(wind_power_watts: float | pd.Series, hours: float = 1.0) -> float | pd.Series:
    """Convert power to energy in kWh using a given time in hours."""
    return np.asarray(wind_power_watts, dtype=float) * hours / 1000.0


def compute_power_metrics(df: pd.DataFrame, wind_speed_col: str = "wind_speed") -> pd.DataFrame:
    result = df.copy()
    result["wind_power_watts"] = calculate_wind_power(result[wind_speed_col])
    result["energy_kwh"] = calculate_energy_output(result["wind_power_watts"], hours=1.0)
    return result
