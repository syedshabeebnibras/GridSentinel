"""Per-node, per-window features for the predictive failure model.

For each (node, prediction_time) pair, we build a feature vector summarizing
the prior `lookback_hours` of telemetry and label it with whether the node
suffered a critical event in the *next* `forecast_hours`.

Why per-node-per-window: nodes are the operational unit ops teams act on,
and a sliding window gives the model thousands of training rows from a single
fleet-week — enough to learn meaningful patterns without a separate dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Each value here is a (kind, label_suffix) pair for the rolling-count features.
_COUNT_KINDS = [
    "thermal_throttle",
    "ecc_uncorrectable",
    "nvlink_fault",
    "pcie_error",
    "network_flap",
    "psu_trip",
    "power_event",
    "cooling_failure",
]


def build_training_set(
    enriched_events: pd.DataFrame,
    util_df: pd.DataFrame,
    fleet: pd.DataFrame,
    ticks_per_hour: int = 12,
    lookback_hours: int = 48,
    forecast_hours: int = 24,
    step_hours: int = 6,
) -> pd.DataFrame:
    """Slide a window over the simulation and assemble a labelled dataset.

    Returns one row per (node, window_end_tick) with:
      - X features: rolling counts per event-kind, severity mix, util stats,
        topology context (zone-level events in lookback window)
      - y label: `failed_next_24h` = any critical event on this node in
        (window_end_tick, window_end_tick + forecast_hours]
    """
    if enriched_events.empty or util_df.empty:
        return pd.DataFrame()

    max_tick = int(max(enriched_events["tick"].max(), util_df["tick"].max()))
    lookback_ticks = lookback_hours * ticks_per_hour
    forecast_ticks = forecast_hours * ticks_per_hour
    step_ticks = step_hours * ticks_per_hour

    nodes = fleet[["node_id", "rack_id", "zone_id"]].copy()
    util_hourly = (
        util_df.assign(node_id=util_df["gpu_id"].str.split("/").str[0])
        .groupby(["node_id", "tick"])["util"]
        .mean()
        .reset_index()
    )
    # zone-level event count in lookback (for topology context feature)
    zone_events_per_tick = (
        enriched_events.groupby(["zone_id", "tick"]).size().reset_index(name="n")
    )

    rows: list[dict] = []
    window_ends = range(lookback_ticks, max_tick - forecast_ticks + 1, step_ticks)
    for window_end in window_ends:
        win_start = window_end - lookback_ticks
        target_end = window_end + forecast_ticks

        win_events = enriched_events[
            (enriched_events["tick"] >= win_start) & (enriched_events["tick"] < window_end)
        ]
        target_events = enriched_events[
            (enriched_events["tick"] >= window_end) & (enriched_events["tick"] < target_end)
        ]
        # Predictive-maintenance scope: COMPONENT-level critical events only.
        # Cascade events (cooling/PSU/power) are stochastic facility incidents
        # not predictable from per-node telemetry, so we exclude them from the
        # label to avoid teaching the model an unlearnable pattern.
        component_kinds = ["ecc_uncorrectable", "nvlink_fault", "pcie_error"]
        crit_targets = (
            target_events.loc[
                (target_events["severity"] == "critical")
                & (target_events["kind"].isin(component_kinds))
            ]
            .groupby("node_id")
            .size()
            .reset_index(name="crit_count")
        )

        # rolling per-node features
        per_node = (
            win_events.groupby(["node_id", "kind"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=_COUNT_KINDS, fill_value=0)
            .add_prefix("n_")
            .reset_index()
        )
        sev_mix = (
            win_events.groupby(["node_id", "severity"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["info", "warn", "critical"], fill_value=0)
            .rename(columns=lambda c: f"sev_{c}")
            .reset_index()
        )

        # Kind × severity cross-tab — lets the model learn that an NVLink WARN
        # means something very different from a thermal_throttle WARN.
        kind_sev = (
            win_events.assign(ks=win_events["kind"] + "_" + win_events["severity"])
            .groupby(["node_id", "ks"])
            .size()
            .unstack(fill_value=0)
        )
        precursor_cols = [
            "ecc_uncorrectable_info",  # corrected ECCs (precursor to uncorrectable)
            "nvlink_fault_warn",       # NVLink warns (precursor to critical NVLink)
            "pcie_error_warn",         # PCIe warns (precursor to critical PCIe)
        ]
        for col in precursor_cols:
            if col not in kind_sev.columns:
                kind_sev[col] = 0
        kind_sev = kind_sev[precursor_cols].reset_index()

        util_win = util_hourly[
            (util_hourly["tick"] >= win_start) & (util_hourly["tick"] < window_end)
        ]
        util_stats = (
            util_win.groupby("node_id")["util"]
            .agg(util_mean="mean", util_max="max", util_std="std")
            .reset_index()
            .fillna({"util_std": 0.0})
        )

        # zone-level pressure
        zone_pressure = (
            zone_events_per_tick[
                (zone_events_per_tick["tick"] >= win_start)
                & (zone_events_per_tick["tick"] < window_end)
            ]
            .groupby("zone_id")["n"]
            .sum()
            .reset_index(name="zone_events_in_window")
        )

        # rack-level thermal stress (count of warn+ thermal events on the rack).
        # win_events already carries rack_id from topology enrichment.
        rack_stress_src = win_events[
            (win_events["kind"] == "thermal_throttle")
            & (win_events["severity"].isin(["warn", "critical"]))
            & (win_events["rack_id"].notna())
        ]
        if rack_stress_src.empty:
            rack_stress = pd.DataFrame({"rack_id": [], "rack_thermal_stress": []})
        else:
            rack_stress = (
                rack_stress_src.groupby("rack_id").size().reset_index(name="rack_thermal_stress")
            )

        # assemble row per node
        df = nodes.merge(per_node, on="node_id", how="left")
        df = df.merge(sev_mix, on="node_id", how="left")
        df = df.merge(kind_sev, on="node_id", how="left")
        df = df.merge(util_stats, on="node_id", how="left")
        df = df.merge(zone_pressure, on="zone_id", how="left")
        df = df.merge(rack_stress, on="rack_id", how="left")
        df = df.merge(crit_targets, on="node_id", how="left")
        df["failed_next_24h"] = (df["crit_count"].fillna(0) > 0).astype(int)
        df["window_end_tick"] = window_end
        df["window_age_days"] = window_end / (ticks_per_hour * 24)
        df = df.drop(columns=["crit_count"]).fillna(0)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


_FEATURE_COLUMNS: tuple[str, ...] = (
    *[f"n_{k}" for k in _COUNT_KINDS],
    "sev_info",
    "sev_warn",
    "sev_critical",
    "ecc_uncorrectable_info",
    "nvlink_fault_warn",
    "pcie_error_warn",
    "util_mean",
    "util_max",
    "util_std",
    "zone_events_in_window",
    "rack_thermal_stress",
    "window_age_days",
)


def feature_columns() -> tuple[str, ...]:
    return _FEATURE_COLUMNS


def X_y(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Slice training dataset into model-ready arrays."""
    X = dataset[list(_FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = dataset["failed_next_24h"].to_numpy(dtype=int)
    return X, y
