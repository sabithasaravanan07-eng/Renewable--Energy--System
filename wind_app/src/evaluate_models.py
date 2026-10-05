from __future__ import annotations

import pandas as pd

from src.train_models import train_and_compare_models


def evaluate_all_models(df: pd.DataFrame, target_column: str = "wind_speed") -> dict:
    results, _, best_model = train_and_compare_models(df, target_column=target_column)
    return {"results": results, "best_model": best_model}
