"""Predictive failure model: per-node 24-hour critical-event classifier.

We use HistGradientBoostingClassifier (the modern sklearn GBM):
  - handles class imbalance well via sample_weight
  - tolerates NaN / missing features natively
  - 10-100x faster than the classic GradientBoosting

Evaluation uses a TIME-BASED split — train on the first 70% of windows by
`window_end_tick`, test on the last 30%. Random splits would leak future
information into training (a known classical mistake in time-series ML).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from gridsentinel.predict.features import X_y, feature_columns


@dataclass
class FailureModel:
    estimator: HistGradientBoostingClassifier
    feature_names: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


def time_based_split(
    dataset: pd.DataFrame, test_fraction: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = dataset["window_end_tick"].quantile(1 - test_fraction)
    train = dataset[dataset["window_end_tick"] <= cutoff]
    test = dataset[dataset["window_end_tick"] > cutoff]
    return train, test


def train(dataset: pd.DataFrame, test_fraction: float = 0.3) -> FailureModel:
    """Fit on time-based train split; report metrics on the held-out tail."""
    train_df, test_df = time_based_split(dataset, test_fraction=test_fraction)
    X_train, y_train = X_y(train_df)
    X_test, y_test = X_y(test_df)

    # class imbalance is severe — weight positives up by their inverse frequency
    pos_rate = max(y_train.mean(), 1e-6)
    weights = np.where(y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate + 1e-6))

    est = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    est.fit(X_train, y_train, sample_weight=weights)

    metrics: dict[str, Any] = {
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    if len(np.unique(y_test)) > 1:
        proba = est.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, proba))
        # precision/recall at the threshold that maximises F1 on the test set
        prec, rec, thr = precision_recall_curve(y_test, proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best = int(np.nanargmax(f1[:-1]))  # ignore last (threshold-less) point
        metrics["best_threshold"] = float(thr[best])
        metrics["precision_at_best_f1"] = float(prec[best])
        metrics["recall_at_best_f1"] = float(rec[best])
        metrics["f1_at_best_f1"] = float(f1[best])
        # Precision@K — what ops actually cares about: of the K riskiest
        # nodes the model flags, how many genuinely fail in the next window?
        for k in (10, 25, 50):
            if len(proba) >= k:
                top_k_idx = np.argsort(proba)[-k:]
                metrics[f"precision_at_{k}"] = float(y_test[top_k_idx].mean())
        metrics["lift_at_10"] = (
            metrics.get("precision_at_10", 0.0) / max(metrics["test_pos_rate"], 1e-6)
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # permutation importance on test set — slower than gain importance, but
    # more honest because it doesn't favour high-cardinality features.
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        try:
            imp = permutation_importance(est, X_test, y_test, n_repeats=3, random_state=42)
            fi_df = pd.DataFrame(
                {
                    "feature": list(feature_columns()),
                    "importance": imp.importances_mean,
                }
            ).sort_values("importance", ascending=False)
        except Exception:
            fi_df = pd.DataFrame({"feature": list(feature_columns()), "importance": 0.0})
    else:
        fi_df = pd.DataFrame({"feature": list(feature_columns()), "importance": 0.0})

    return FailureModel(
        estimator=est,
        feature_names=feature_columns(),
        metrics=metrics,
        feature_importance=fi_df,
    )


def top_at_risk(
    model: FailureModel,
    latest_window: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Score the most-recent window of features → top N highest-risk nodes."""
    if latest_window.empty:
        return pd.DataFrame()
    X, _ = X_y(latest_window)
    proba = model.predict_proba(X)
    out = latest_window[["node_id", "rack_id", "zone_id"]].copy()
    out["failure_risk_24h"] = proba
    return out.sort_values("failure_risk_24h", ascending=False).head(n).reset_index(drop=True)
