"""Featurize each incident for clustering.

Features mix structural (what failed, where) and temporal (when, how long)
signals. The structural ones drive the cluster split; the temporal ones help
distinguish recurring patterns (e.g. "afternoon thermal bursts on a specific
rack" vs "midnight cooling cascades").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SEVERITY_ORDINAL = {"info": 0, "warn": 1, "critical": 2}
_KIND_VOCAB = (
    "thermal_throttle",
    "ecc_uncorrectable",
    "nvlink_fault",
    "pcie_error",
    "psu_trip",
    "network_flap",
    "cooling_failure",
    "power_event",
)
_SCOPE_VOCAB = ("gpu", "node", "rack", "zone", "feed", "spine")


def featurize(incidents: pd.DataFrame, ticks_per_hour: int = 12) -> pd.DataFrame:
    """Return a feature DataFrame aligned 1:1 with `incidents`.

    Numeric columns only — directly usable by HDBSCAN / KMeans after scaling.
    """
    if incidents.empty:
        return pd.DataFrame()

    feats: dict[str, np.ndarray] = {}

    # one-hot kind
    for k in _KIND_VOCAB:
        feats[f"kind_{k}"] = (incidents["root_kind"] == k).astype(float).to_numpy()

    # one-hot scope
    for s in _SCOPE_VOCAB:
        feats[f"scope_{s}"] = (incidents["scope"] == s).astype(float).to_numpy()

    # log size + duration
    feats["log_member_count"] = np.log1p(incidents["member_count"].to_numpy())
    feats["duration_ticks"] = incidents["duration_ticks"].fillna(0).to_numpy(dtype=float)

    # severity ordinal
    feats["severity"] = (
        incidents["severity_max"].map(_SEVERITY_ORDINAL).fillna(0).to_numpy(dtype=float)
    )

    # hour-of-day cyclical encoding
    hours_per_day = 24
    hod = (incidents["opened_tick"] // ticks_per_hour) % hours_per_day
    feats["hod_sin"] = np.sin(2 * np.pi * hod / hours_per_day).to_numpy()
    feats["hod_cos"] = np.cos(2 * np.pi * hod / hours_per_day).to_numpy()

    # benign flag (all members benign → likely noise cluster)
    feats["all_benign"] = incidents["all_benign"].astype(float).to_numpy()

    return pd.DataFrame(feats, index=incidents.index)
