"""Transparent, rule-based environmental indicators (not an ML model)."""


def predict_weather(record: dict) -> dict:
    cloud = record["cloud_cover_percent"]
    humidity = record["humidity_percent"]
    wind = record["wind_kmh"]

    if cloud < 20:
        sky = "Clear"
    elif cloud < 60:
        sky = "Partly cloudy"
    else:
        sky = "Overcast"

    if humidity > 80 and cloud > 50:
        note = "High humidity and cloud cover; expect reduced solar yield."
    elif wind > 40:
        note = "Strong wind; panels should still perform normally."
    else:
        note = "Conditions are favourable for solar generation."

    return {
        "sky_condition": sky,
        "note": note,
    }
