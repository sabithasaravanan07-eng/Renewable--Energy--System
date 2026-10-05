"""E = P x t. Power in watts, time in hours, energy in Wh and kWh."""


def estimate_energy(power_w: float, hours: float) -> dict:
    energy_wh = power_w * hours
    return {
        "energy_wh": energy_wh,
        "energy_kwh": energy_wh / 1000.0,
    }
