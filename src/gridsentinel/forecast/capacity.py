"""Convert utilization forecasts into capacity-headroom answers."""
from __future__ import annotations

import pandas as pd


def hours_to_threshold(
    forecast_df: pd.DataFrame,
    max_gpu_hours_per_hour: float,
    threshold: float = 0.85,
) -> int | None:
    """Hours until forecast crosses `threshold × capacity`.

    Returns None if the forecast never crosses within the horizon.
    """
    limit = threshold * max_gpu_hours_per_hour
    over = forecast_df.loc[forecast_df["yhat"] >= limit]
    if over.empty:
        return None
    first = int(over.iloc[0]["tick"])
    base = int(forecast_df.iloc[0]["tick"])
    return max(0, first - base)


def weeks_to_exhaustion(
    forecast_df: pd.DataFrame,
    max_gpu_hours_per_hour: float,
    threshold: float = 0.85,
) -> float | None:
    h = hours_to_threshold(forecast_df, max_gpu_hours_per_hour, threshold)
    return None if h is None else round(h / (24 * 7), 2)


def headroom_pct(current_gpu_hours: float, max_gpu_hours: float) -> float:
    if max_gpu_hours <= 0:
        return 0.0
    return max(0.0, 1.0 - current_gpu_hours / max_gpu_hours)
