"""Unsupervised anomaly detection — the 'defense in depth' layer.

The supervised classifier only catches failure modes it was trained on.
IsolationForest learns the *shape* of normal telemetry and flags any window
that doesn't look normal — including failure modes that never appeared in
training. Real ops platforms (Datadog Watchdog, Honeycomb BubbleUp,
AWS Lookout for Equipment) all include this layer.

Score interpretation: lower = more anomalous (sklearn convention). We invert
to "anomaly score in [0, 1]" for dashboard readability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyModel:
    estimator: IsolationForest
    scaler: StandardScaler
    feature_names: list[str]
    contamination: float

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly score in [0, 1] (1 = most anomalous)."""
        Xs = self.scaler.transform(X[self.feature_names].to_numpy())
        # decision_function: higher = more normal. Flip + normalize to [0,1].
        raw = -self.estimator.decision_function(Xs)
        # min-max normalize to [0, 1] using a stable scaling
        lo, hi = np.percentile(raw, [5, 95])
        return np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1)


def train_anomaly_detector(
    ts_dataset: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    test_fraction: float = 0.3,
) -> AnomalyModel:
    """Fit IsolationForest on the *normal* (non-failing) windows from the
    training tail, then return a model that scores any new window.

    Why train only on non-failing windows: IsolationForest is unsupervised but
    fitting on all data lets failures pollute the "normal" manifold. Training
    on the negative class gives a cleaner detection boundary.
    """
    cutoff = ts_dataset["window_end_tick"].quantile(1 - test_fraction)
    train_df = ts_dataset[
        (ts_dataset["window_end_tick"] <= cutoff) & (ts_dataset["failed_next_24h"] == 0)
    ]
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    est = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    est.fit(X_train_s)
    return AnomalyModel(
        estimator=est,
        scaler=scaler,
        feature_names=feature_cols,
        contamination=contamination,
    )


def evaluate_anomaly(model: AnomalyModel, ts_dataset: pd.DataFrame) -> dict[str, float]:
    """Sanity-check: do anomaly scores correlate with the supervised label?
    Not a primary metric — but a useful sanity check that the detector isn't
    purely random."""
    from sklearn.metrics import roc_auc_score

    scores = model.score(ts_dataset)
    y = ts_dataset["failed_next_24h"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"score_auc_vs_label": float("nan")}
    return {
        "score_auc_vs_label": float(roc_auc_score(y, scores)),
        "score_mean_positive": float(scores[y == 1].mean()),
        "score_mean_negative": float(scores[y == 0].mean()),
    }
