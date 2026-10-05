"""P = A x G x eta x r (see README equations)."""

PERFORMANCE_RATIO = 0.90


def calculate_power(panel_area: float, effective_irradiance_w_m2: float, efficiency: float,
                     performance_ratio: float = PERFORMANCE_RATIO) -> float:
    return max(panel_area * effective_irradiance_w_m2 * efficiency * performance_ratio, 0.0)
