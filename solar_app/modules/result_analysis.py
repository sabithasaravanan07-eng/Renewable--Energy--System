"""Human-readable observations plus a Chart.js-ready payload."""


def analyze_result(record: dict, predicted_power: float, energy: dict, metrics: dict, solar: dict) -> dict:
    observations = []

    if solar["cloud_attenuation_factor"] < 0.6:
        observations.append("Heavy cloud cover is significantly attenuating irradiance.")
    if solar["temperature_factor"] < 0.95:
        observations.append("High ambient temperature is reducing panel output slightly.")
    if predicted_power <= 0:
        observations.append("Predicted output is at or near zero under these conditions.")
    if not observations:
        observations.append("Conditions are within a typical operating range for this system.")

    chart_data = {
        "labels": ["Raw irradiance", "Effective irradiance"],
        "values": [solar["raw_irradiance_w_m2"], solar["effective_irradiance_w_m2"]],
        "predicted_power_w": round(predicted_power, 2),
        "energy_kwh": round(energy["energy_kwh"], 2),
        "model_r2": metrics.get("r2"),
    }

    return {
        "observations": observations,
        "chart_data": chart_data,
    }
