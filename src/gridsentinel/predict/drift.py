"""Feature drift detection using Population Stability Index (PSI).

PSI is the industry-standard metric (banking, ad-tech, ops ML) for detecting
distribution shift between two cohorts. Convention:

  PSI < 0.10 → no meaningful drift
  0.10–0.25 → moderate drift, monitor / consider retraining
  PSI > 0.25 → significant drift, retrain

Computed per feature by binning the baseline distribution into deciles and
measuring how the current distribution redistributes across those bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_quantiles(df: pd.DataFrame, feature_cols: list[str], n_bins: int = 10) -> pd.DataFrame:
    """Compute baseline decile edges per feature; rows = features, cols = edges.
    Saved at training time, reused at scoring time."""
    rows = []
    for col in feature_cols:
        edges = np.quantile(df[col].to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1))
        # ensure strictly increasing — pad tiny gaps
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        rows.append({"feature": col, **{f"q{i}": float(edges[i]) for i in range(len(edges))}})
    return pd.DataFrame(rows)


def psi(reference: np.ndarray, current: np.ndarray, edges: np.ndarray) -> float:
    """Population Stability Index between two samples using fixed bin edges."""
    eps = 1e-6
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)
    ref_p = np.clip(ref_p, eps, None)
    cur_p = np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-feature PSI against the registered baseline distribution."""
    rows = []
    edge_cols = [c for c in edges_df.columns if c.startswith("q")]
    for _, row in edges_df.iterrows():
        feat = row["feature"]
        edges = np.array([row[c] for c in edge_cols])
        if feat not in baseline_df.columns or feat not in current_df.columns:
            continue
        score = psi(
            baseline_df[feat].to_numpy(dtype=float),
            current_df[feat].to_numpy(dtype=float),
            edges,
        )
        rows.append(
            {
                "feature": feat,
                "psi": score,
                "severity": (
                    "stable" if score < 0.10
                    else "moderate" if score < 0.25
                    else "significant"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
