"""Forecast aggregate GPU-hours used per period.

For the demo we forecast HOURLY GPU-hours (the simulator only produces 1-2
days of data, so daily forecasting needs more history). The model is
SARIMAX(1,1,1)(1,0,1,24) — captures diurnal seasonality.

With longer simulations (≥ 14 days), switch the series to daily aggregation
and seasonal_order=(1,1,1,7) for weekly seasonality.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# statsmodels emits a flood of convergence warnings on short series — silence
# them so the dashboard caption stays clean. The wider CI band already signals
# model uncertainty visually.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")


def hourly_gpu_hours(util_df: pd.DataFrame) -> pd.Series:
    """Sum of utilization per tick → GPU-hours used in that hour."""
    if util_df.empty:
        return pd.Series(dtype=float)
    s = util_df.groupby("tick")["util"].sum().sort_index()
    s.index.name = "tick"
    return s


def forecast_gpu_hours(
    hourly_series: pd.Series,
    horizon_hours: int = 24,
    ticks_per_hour: int = 12,
) -> pd.DataFrame:
    """Decomposition model: linear daily trend + repeating diurnal pattern.

    Why not SARIMAX: with short histories (< 30 days) SARIMAX mean-reverts on
    the trend component, hiding real growth. A decomposition-style model with
    explicit linear trend + repeating diurnal is more honest and far more
    interpretable for an ops audience.

    `hourly_series.index` is in raw ticks (units of `ticks_per_hour` per hour).
    """
    if len(hourly_series) < 24:
        mean = float(hourly_series.mean()) if len(hourly_series) else 0.0
        std = float(hourly_series.std()) if len(hourly_series) > 1 else mean * 0.1
        last = int(hourly_series.index[-1]) if len(hourly_series) else 0
        idx = np.arange(last + 1, last + 1 + horizon_hours)
        return pd.DataFrame(
            {
                "tick": idx,
                "yhat": [mean] * horizon_hours,
                "yhat_lower": [mean - 1.28 * std] * horizon_hours,
                "yhat_upper": [mean + 1.28 * std] * horizon_hours,
            }
        )

    ticks_per_day = ticks_per_hour * 24
    ticks = np.asarray(hourly_series.index, dtype=float)
    y = np.asarray(hourly_series.to_numpy(), dtype=float)
    hod = ((ticks // ticks_per_hour) % 24).astype(int)
    day = (ticks // ticks_per_day).astype(int)

    # 1) Daily means → fit linear trend over days
    daily_df = pd.DataFrame({"day": day, "y": y}).groupby("day")["y"].mean()
    days = daily_df.index.to_numpy(dtype=float)
    slope_per_day, intercept = np.polyfit(days, daily_df.to_numpy(), 1)
    slope_per_tick = slope_per_day / ticks_per_day

    # 2) Diurnal residual: average residual per hour-of-day
    trend = slope_per_tick * ticks + intercept
    residual = y - trend
    diurnal = np.array(
        [residual[hod == h].mean() if (hod == h).any() else 0.0 for h in range(24)]
    )

    # 3) Residual std for CI
    sigma = float((y - trend - diurnal[hod]).std()) if len(y) > 1 else 0.0

    start_tick = int(hourly_series.index[-1]) + ticks_per_hour
    future_ticks = np.arange(start_tick, start_tick + horizon_hours * ticks_per_hour, ticks_per_hour)
    future_hod = ((future_ticks // ticks_per_hour) % 24).astype(int)
    yhat = slope_per_tick * future_ticks + intercept + diurnal[future_hod]
    yhat = np.clip(yhat, 0, None)

    return pd.DataFrame(
        {
            "tick": future_ticks,
            "yhat": yhat,
            "yhat_lower": np.clip(yhat - 1.28 * sigma, 0, None),
            "yhat_upper": yhat + 1.28 * sigma,
        }
    )
