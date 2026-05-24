"""Time-series feature extraction from continuous DCGM metrics.

For each (node, window) we summarise the prior `lookback_hours` of raw
telemetry into a numeric vector. tsfresh-style features without the tsfresh
dependency — keeps the model fast and easy to reason about.

Per signal (gpu_temp_c, gpu_power_w, gpu_sm_util, ecc_corrected_rate,
nvlink_crc_rate, pcie_aer_rate) we extract:

  - mean, std, min, max, range
  - linear trend slope
  - lag-1 autocorrelation (persistence)
  - spike count (samples > μ + 2σ within window)
  - last-value
  - max-of-rolling-std (volatility regime detector)

For the counter-derived rates, we take np.diff(counter_total) per GPU
first — counters are monotonic, so absolute values are useless; rates are
the predictive signal.

Aggregation: signals are per-GPU. We aggregate to node-level by taking the
max across the 8 GPUs (worst-GPU-on-node — the alert that matters).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SIGNALS = ("gpu_temp_c", "gpu_power_w", "gpu_sm_util")
_COUNTERS = ("ecc_corrected_total", "nvlink_crc_total", "pcie_aer_total")


def _summarise(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "min", "max", "range", "slope",
                                  "ac1", "spike_count", "last", "vol_max")}
    arr = arr.astype(float)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std())
    minv = float(arr.min())
    maxv = float(arr.max())
    rng = maxv - minv
    if n >= 2:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
        # lag-1 autocorrelation
        a, b = arr[:-1] - arr[:-1].mean(), arr[1:] - arr[1:].mean()
        denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-12
        ac1 = float((a * b).sum() / denom)
    else:
        slope = 0.0
        ac1 = 0.0
    spike_threshold = mean + 2 * std
    spike_count = int((arr > spike_threshold).sum())
    last = float(arr[-1])
    # rolling-std volatility regime
    if n >= 6:
        win = max(3, n // 6)
        rolling_std = pd.Series(arr).rolling(win).std().to_numpy()
        vol_max = float(np.nanmax(rolling_std)) if not np.isnan(rolling_std).all() else 0.0
    else:
        vol_max = std
    return {
        "mean": mean, "std": std, "min": minv, "max": maxv, "range": rng,
        "slope": slope, "ac1": ac1, "spike_count": float(spike_count),
        "last": last, "vol_max": vol_max,
    }


def extract_window_features(metrics_window: pd.DataFrame) -> pd.DataFrame:
    """Input: rows from data/synthetic/metrics.parquet limited to a single
    time window. Output: one row per node with all summarised features.
    """
    if metrics_window.empty:
        return pd.DataFrame()

    # Convert monotonic counters to per-sample rates per GPU.
    w = metrics_window.sort_values(["gpu_id", "tick"]).copy()
    for col in _COUNTERS:
        rate_col = col.replace("_total", "_rate")
        w[rate_col] = w.groupby("gpu_id")[col].diff().fillna(0).clip(lower=0)

    rate_cols = tuple(c.replace("_total", "_rate") for c in _COUNTERS)
    signals = _SIGNALS + rate_cols

    # Summarise per GPU, then aggregate to node = MAX of summary across GPUs
    # (worst-GPU-on-node — that's what triggers a real page).
    feature_rows: list[dict] = []
    for node_id, node_group in w.groupby("node_id"):
        row: dict[str, float | str] = {"node_id": str(node_id)}
        for sig in signals:
            per_gpu = []
            for _, gpu_group in node_group.groupby("gpu_id"):
                per_gpu.append(_summarise(gpu_group[sig].to_numpy()))
            if not per_gpu:
                continue
            for key in per_gpu[0]:
                row[f"{sig}__{key}"] = max(d[key] for d in per_gpu)
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


_PRECURSOR_KINDS = ("ecc_uncorrectable", "nvlink_fault", "pcie_error", "thermal_throttle")


def _event_count_features(win_events: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """V1-style event-count features that complement the continuous-telemetry ones.
    Particularly: per-(kind, severity) precursor counts that the supervised
    model can latch onto directly. These coexist with the rolling-stat ts
    features — the model sees both and uses whichever has signal."""
    if win_events.empty:
        return pd.DataFrame()
    n = win_events.assign(
        ks=win_events["kind"].astype(str) + "_" + win_events["severity"].astype(str)
    )
    counts = (
        n.groupby(["node_id", "ks"]).size().unstack(fill_value=0)
    )
    keep = []
    for k in _PRECURSOR_KINDS:
        for sev in ("info", "warn", "critical"):
            col = f"{k}_{sev}"
            if col not in counts.columns:
                counts[col] = 0
            keep.append(col)
    counts = counts[keep].add_prefix("ev_").reset_index()
    return counts


def build_timeseries_dataset(
    metrics_df: pd.DataFrame,
    enriched_events: pd.DataFrame,
    fleet: pd.DataFrame,
    ticks_per_hour: int = 12,
    lookback_hours: int = 48,
    forecast_hours: int = 24,
    step_hours: int = 6,
) -> pd.DataFrame:
    """Slide a window over the continuous metrics, extract HYBRID features
    (continuous-telemetry rolling stats + per-(kind,severity) event counts),
    label as failure-in-next-24h for component-level critical events.

    Cascade events (cooling/PSU/power) excluded from the label — those are
    facility incidents not predictable from per-node telemetry.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    max_tick = int(max(metrics_df["tick"].max(), enriched_events["tick"].max()))
    lookback = lookback_hours * ticks_per_hour
    horizon = forecast_hours * ticks_per_hour
    step = step_hours * ticks_per_hour

    component_kinds = ["ecc_uncorrectable", "nvlink_fault", "pcie_error"]

    out_rows: list[pd.DataFrame] = []
    for window_end in range(lookback, max_tick - horizon + 1, step):
        win_start = window_end - lookback
        target_end = window_end + horizon

        m_win = metrics_df[(metrics_df["tick"] >= win_start) & (metrics_df["tick"] < window_end)]
        if m_win.empty:
            continue
        feats = extract_window_features(m_win)
        feats["window_end_tick"] = window_end

        # Hybrid: also blend in event-count features
        ev_win = enriched_events[
            (enriched_events["tick"] >= win_start) & (enriched_events["tick"] < window_end)
        ]
        ev_feats = _event_count_features(ev_win, fleet)
        if not ev_feats.empty:
            feats = feats.merge(ev_feats, on="node_id", how="left")

        crit_targets = (
            enriched_events.loc[
                (enriched_events["tick"] >= window_end)
                & (enriched_events["tick"] < target_end)
                & (enriched_events["severity"] == "critical")
                & (enriched_events["kind"].isin(component_kinds))
            ]
            .groupby("node_id")
            .size()
            .reset_index(name="crit_count")
        )
        feats = feats.merge(crit_targets, on="node_id", how="left")
        feats["failed_next_24h"] = (feats["crit_count"].fillna(0) > 0).astype(int)
        feats = feats.drop(columns=["crit_count"])
        out_rows.append(feats)

    if not out_rows:
        return pd.DataFrame()

    dataset = pd.concat(out_rows, ignore_index=True).fillna(0)
    dataset = dataset.merge(fleet[["node_id", "rack_id", "zone_id"]], on="node_id", how="left")
    return dataset


def ts_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """All non-metadata, non-label columns."""
    excluded = {"node_id", "rack_id", "zone_id", "window_end_tick", "failed_next_24h"}
    return [c for c in dataset.columns if c not in excluded]
