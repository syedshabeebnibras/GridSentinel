"""Cox Proportional Hazards survival analysis.

Models *time-to-next-critical-event* per node as a function of the time-
series features. Useful in PdM for two reasons over a binary classifier:

  1. Outputs a hazard function over time — "this node's risk *next hour* vs
     *next day*", not just "yes/no in 24h."
  2. Handles right-censored observations cleanly — nodes that haven't failed
     yet still inform the model.

We report Harrell's concordance index (C-index) — the survival-analysis
analog of ROC AUC. Industry standard is 0.65-0.75 on real telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SurvivalModel:
    estimator: object  # lifelines.CoxPHFitter
    feature_names: list[str]
    metrics: dict[str, float]
    baseline_survival: pd.DataFrame  # baseline survival curve

    def predict_partial_hazard(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_partial_hazard(X).to_numpy()

    def predict_survival_function(self, X: pd.DataFrame, times: list[float] | None = None):
        return self.estimator.predict_survival_function(X, times=times)


def build_survival_dataset(
    ts_dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon_hours: int = 24,
    ticks_per_hour: int = 12,
) -> pd.DataFrame:
    """Convert the (per-node, per-window) classification dataset into a
    survival dataset.

      - `duration` = hours until end of window's forecast horizon
      - `event` = 1 if failed within horizon, 0 if censored
    """
    if ts_dataset.empty:
        return pd.DataFrame()
    df = ts_dataset[feature_cols + ["failed_next_24h"]].copy()
    df["duration"] = float(horizon_hours)
    df["event"] = df["failed_next_24h"].astype(int)
    df = df.drop(columns=["failed_next_24h"])
    return df


def train_survival(
    ts_dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon_hours: int = 24,
    test_fraction: float = 0.3,
) -> SurvivalModel | None:
    """Fit Cox PH; return SurvivalModel or None if lifelines refuses to fit
    (e.g. perfectly separable features → singular matrix)."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        return None

    surv_df = build_survival_dataset(ts_dataset, feature_cols, horizon_hours=horizon_hours)
    if surv_df.empty:
        return None

    # Trim near-constant columns — Cox PH fails on those.
    keep = [c for c in feature_cols if surv_df[c].std() > 1e-6]
    if not keep:
        return None
    surv_df = surv_df[keep + ["duration", "event"]].copy()

    cutoff = ts_dataset["window_end_tick"].quantile(1 - test_fraction)
    train_mask = ts_dataset["window_end_tick"] <= cutoff
    train_df = surv_df.loc[train_mask.values].reset_index(drop=True)
    test_df = surv_df.loc[(~train_mask).values].reset_index(drop=True)

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    try:
        cph.fit(train_df, duration_col="duration", event_col="event", show_progress=False)
    except Exception as e:
        # Lifelines occasionally rejects ill-conditioned data — bail out cleanly.
        return SurvivalModel(
            estimator=None,
            feature_names=keep,
            metrics={"error": str(e)[:200]},
            baseline_survival=pd.DataFrame(),
        )

    test_partial = cph.predict_partial_hazard(test_df[keep])
    c_index = (
        float(
            concordance_index(
                test_df["duration"],
                -test_partial,  # negative because higher hazard → shorter survival
                test_df["event"],
            )
        )
        if len(test_df) > 0
        else float("nan")
    )
    metrics = {
        "c_index": c_index,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "event_rate_train": float(train_df["event"].mean()),
        "event_rate_test": float(test_df["event"].mean()),
        "concordance_train": float(cph.concordance_index_),
    }
    return SurvivalModel(
        estimator=cph,
        feature_names=keep,
        metrics=metrics,
        baseline_survival=cph.baseline_survival_,
    )
