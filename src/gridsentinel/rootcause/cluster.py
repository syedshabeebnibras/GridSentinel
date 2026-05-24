"""Cluster incidents into recurring root-cause families.

HDBSCAN handles variable density and noise without requiring `k`. We scale
features first so one-hots and log-counts don't blow each other out, then
label each cluster by majority kind/scope + the dominant time-of-day.

The output is two tables:
  - incidents augmented with `cluster_id` and `cluster_label`
  - a recurring-families summary ranked by frequency
"""
from __future__ import annotations

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_incidents(
    features: pd.DataFrame,
    incidents: pd.DataFrame,
    min_cluster_size: int = 8,
) -> pd.DataFrame:
    """Returns incidents with new `cluster_id` and `cluster_label` columns.

    cluster_id == -1 means HDBSCAN labelled the row as noise.
    """
    if features.empty:
        return incidents.assign(cluster_id=pd.Series(dtype=int), cluster_label=pd.Series(dtype=str))

    X = StandardScaler().fit_transform(features.to_numpy())
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=False)
    labels = clusterer.fit_predict(X)

    out = incidents.copy()
    out["cluster_id"] = labels
    out["cluster_label"] = _label_clusters(out, ticks_per_hour=12)
    return out


def _label_clusters(incidents: pd.DataFrame, ticks_per_hour: int) -> pd.Series:
    """Build a human-readable label per cluster using majority kind/scope + peak hour."""
    labels = pd.Series(index=incidents.index, dtype="object")
    for cid, group in incidents.groupby("cluster_id"):
        if cid == -1:
            label = "noise / unclustered"
        else:
            kind = group["root_kind"].mode().iat[0]
            scope = group["scope"].mode().iat[0]
            sev = group["severity_max"].mode().iat[0]
            hours = (group["opened_tick"] // ticks_per_hour) % 24
            peak_hour = int(hours.mode().iat[0])
            label = f"{kind} · {scope} · {sev} · ~{peak_hour:02d}:00 ({len(group)}×)"
        labels.loc[group.index] = label
    return labels


def top_recurring(clustered: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N clusters by frequency (excluding noise).   One row per cluster."""
    real = clustered[clustered["cluster_id"] != -1]
    if real.empty:
        return pd.DataFrame(
            columns=["cluster_id", "cluster_label", "count", "critical_count", "all_benign_ratio"]
        )
    agg = (
        real.groupby(["cluster_id", "cluster_label"])
        .agg(
            count=("cluster_id", "size"),
            critical_count=("severity_max", lambda s: (s == "critical").sum()),
            all_benign_ratio=("all_benign", "mean"),
            mean_members=("member_count", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return agg


def cluster_quality(features: pd.DataFrame, cluster_ids: pd.Series) -> float | None:
    """Silhouette score over non-noise points. None if not computable.

    Range [-1, 1]; > 0.25 is decent, > 0.5 is strong. Used as a dashboard
    confidence signal — lets the user know whether the clusters mean anything.
    """
    mask = cluster_ids != -1
    if mask.sum() < 3 or cluster_ids[mask].nunique() < 2:
        return None
    X = StandardScaler().fit_transform(features.loc[mask].to_numpy())
    try:
        return float(silhouette_score(X, cluster_ids[mask].to_numpy()))
    except ValueError:
        return None
