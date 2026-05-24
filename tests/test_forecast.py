import numpy as np
import pandas as pd

from gridsentinel.forecast.capacity import (
    headroom_pct,
    hours_to_threshold,
    weeks_to_exhaustion,
)
from gridsentinel.forecast.utilization import forecast_gpu_hours, hourly_gpu_hours


def _util_df_diurnal(hours: int = 48):
    rows = []
    for h in range(hours):
        diurnal = 0.5 + 0.25 * np.sin((h - 6) * np.pi / 12)
        for g in range(20):
            rows.append({"tick": h, "gpu_id": f"g{g}", "util": float(diurnal)})
    return pd.DataFrame(rows)


def test_forecast_extrapolates_trend():
    """If the input has a clear upward trend, the forecast should continue it."""
    # 14 days × 24 h, growing baseline 0.45 → 0.55
    rows = []
    for d in range(14):
        for h in range(24):
            tick = (d * 24 + h) * 12  # match real sim cadence
            base = 0.45 + 0.008 * d
            util_val = base + 0.25 * np.sin((h - 6) * np.pi / 12)
            for g in range(50):
                rows.append({"tick": tick, "gpu_id": f"g{g}", "util": float(util_val)})
    s = hourly_gpu_hours(pd.DataFrame(rows))
    fc = forecast_gpu_hours(s, horizon_hours=14 * 24, ticks_per_hour=12)
    # forecast peak should be ABOVE the observed peak (continued growth)
    assert fc["yhat"].max() > s.max(), (
        f"expected forecast peak > observed {s.max()}, got {fc['yhat'].max()}"
    )


def test_hourly_gpu_hours_sums_util():
    df = _util_df_diurnal(hours=4)
    s = hourly_gpu_hours(df)
    assert len(s) == 4
    assert s.iloc[0] > 0


def test_forecast_returns_horizon_rows():
    s = hourly_gpu_hours(_util_df_diurnal(hours=48))
    out = forecast_gpu_hours(s, horizon_hours=12, ticks_per_hour=1)
    assert len(out) == 12
    assert {"tick", "yhat", "yhat_lower", "yhat_upper"} <= set(out.columns)
    assert (out["yhat_upper"] >= out["yhat_lower"]).all()


def test_forecast_fallback_for_tiny_series():
    s = pd.Series([10.0, 11.0, 12.0], index=[0, 1, 2])
    out = forecast_gpu_hours(s, horizon_hours=5)
    assert len(out) == 5


def test_headroom_pct_basic():
    assert headroom_pct(0, 100) == 1.0
    assert headroom_pct(50, 100) == 0.5
    assert headroom_pct(150, 100) == 0.0
    assert headroom_pct(50, 0) == 0.0


def test_weeks_to_exhaustion_returns_none_when_no_crossing():
    fc = pd.DataFrame({"tick": range(24), "yhat": [10.0] * 24})
    assert weeks_to_exhaustion(fc, max_gpu_hours_per_hour=100) is None


def test_hours_to_threshold_finds_crossing():
    fc = pd.DataFrame({"tick": range(10), "yhat": [50.0] * 5 + [90.0] * 5})
    h = hours_to_threshold(fc, max_gpu_hours_per_hour=100, threshold=0.85)
    assert h == 5
