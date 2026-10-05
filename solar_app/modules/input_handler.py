"""Parses and validates the raw form input for the current-conditions
power/energy estimate (the /input page). This is a snapshot estimate and
is intentionally separate from the date/time-driven forecast feature."""

FIELDS = {
    "irradiance_w_m2": (0.0, 1500.0, "Irradiance (W/m2)"),
    "temperature_c": (-20.0, 60.0, "Ambient temperature (C)"),
    "humidity_percent": (0.0, 100.0, "Humidity (%)"),
    "wind_kmh": (0.0, 150.0, "Wind speed (km/h)"),
    "cloud_cover_percent": (0.0, 100.0, "Cloud cover (%)"),
    "panel_area": (0.01, 100000.0, "Panel area (m2)"),
    "efficiency": (0.01, 1.0, "Panel efficiency (0-1)"),
    "operating_hours": (0.0, 24.0, "Operating hours"),
}


def parse_and_validate(form):
    """Validate a Flask form dict against FIELDS.

    Returns (values, errors). `values` maps field name -> float, only
    populated when there were no errors.
    """
    errors = []
    values = {}

    for key, (low, high, label) in FIELDS.items():
        raw = (form.get(key) or "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not (low <= value <= high):
            errors.append(f"{label} must be between {low} and {high}.")
            continue
        values[key] = value

    if errors:
        return {}, errors

    return values, []
