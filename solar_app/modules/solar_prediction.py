"""Transparent, rule-based adjustment of raw irradiance into an
'effective' irradiance figure, accounting for cloud cover and a bounded
temperature coefficient around a 25 C reference (see README equations)."""

TEMP_REFERENCE_C = 25.0
TEMP_COEFFICIENT = -0.0045  # per degree C, bounded below
MIN_TEMP_FACTOR = 0.75
MAX_TEMP_FACTOR = 1.05


def predict_solar_conditions(record: dict) -> dict:
    irradiance = record["irradiance_w_m2"]
    cloud_cover = record["cloud_cover_percent"]
    temperature = record["temperature_c"]

    cloud_attenuation = 1 - (cloud_cover / 140.0)
    cloud_attenuation = max(cloud_attenuation, 0.0)

    temp_factor = 1 + TEMP_COEFFICIENT * (temperature - TEMP_REFERENCE_C)
    temp_factor = min(max(temp_factor, MIN_TEMP_FACTOR), MAX_TEMP_FACTOR)

    effective_irradiance = irradiance * cloud_attenuation * temp_factor
    effective_irradiance = max(effective_irradiance, 0.0)

    return {
        "raw_irradiance_w_m2": round(irradiance, 2),
        "cloud_attenuation_factor": round(cloud_attenuation, 4),
        "temperature_factor": round(temp_factor, 4),
        "effective_irradiance_w_m2": round(effective_irradiance, 2),
    }
