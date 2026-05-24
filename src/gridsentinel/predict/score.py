"""Industry-grade PdM pipeline CLI.

Pipeline:
  1. Load continuous metrics + events + fleet
  2. Build time-series feature dataset (rolling windows)
  3. Train calibrated classifier with rolling-origin TSCV
  4. Train Cox PH survival model
  5. Train IsolationForest anomaly detector
  6. Persist drift baseline (per-feature quantiles)
  7. Register all artifacts to data/models/v{N}/
  8. Log everything to MLflow at mlruns/
  9. Print headline metrics + top at-risk nodes
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridsentinel.ingest.telemetry import (
    enrich_with_topology,
    load_events,
    load_fleet,
)
from gridsentinel.predict.anomaly import evaluate_anomaly, train_anomaly_detector
from gridsentinel.predict.calibrated import (
    CalibratedFailureModel,
    train_calibrated,
)
from gridsentinel.predict.drift import baseline_quantiles
from gridsentinel.predict.registry import register
from gridsentinel.predict.survival import train_survival
from gridsentinel.predict.tracking import log_training_run
from gridsentinel.predict.ts_features import (
    build_timeseries_dataset,
    ts_feature_columns,
)

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "predict"


def load_metrics() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[3] / "data" / "synthetic" / "metrics.parquet"
    if not p.exists():
        raise FileNotFoundError(
            "data/synthetic/metrics.parquet not found. Run "
            "`python -m gridsentinel.simulator.emit` first."
        )
    return pd.read_parquet(p)


def top_at_risk(
    cal_model: CalibratedFailureModel,
    latest_window: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    if latest_window.empty:
        return pd.DataFrame()
    proba = cal_model.predict_proba(latest_window)
    out = latest_window[["node_id", "rack_id", "zone_id"]].copy()
    out["failure_risk_24h"] = proba
    return out.sort_values("failure_risk_24h", ascending=False).head(n).reset_index(drop=True)


def run(n_splits: int = 5) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events = load_events()
    fleet = load_fleet()
    metrics = load_metrics()
    enriched = enrich_with_topology(events, fleet)

    print(f"loaded {len(events):,} events, {len(metrics):,} metric samples, {len(fleet)} nodes")
    print("building time-series dataset...")
    ts_dataset = build_timeseries_dataset(metrics, enriched, fleet)
    if ts_dataset.empty:
        print("dataset empty — not enough simulation history")
        return {}
    feature_cols = ts_feature_columns(ts_dataset)
    print(f"  {len(ts_dataset):,} (node, window) rows × {len(feature_cols)} features")

    print("training calibrated classifier (TimeSeriesSplit CV)...")
    cal_model = train_calibrated(ts_dataset, feature_cols, n_splits=n_splits)
    m = cal_model.metrics
    print(
        f"  ROC AUC = {m['roc_auc_mean']:.3f} ± {m['roc_auc_std']:.3f} "
        f"(across {m['n_folds']} folds)"
    )
    print(f"  PR AUC  = {m['pr_auc_mean']:.3f}")
    print(f"  Brier   = {m['brier_mean']:.4f}  (lower = better-calibrated)")
    print(f"  Prec@10 = {m['precision_at_10_mean']:.3f}")
    print(f"  Lift@10 = {m['lift_at_10_mean']:.2f}×")

    print("training Cox PH survival model...")
    surv = train_survival(ts_dataset, feature_cols)
    if surv is None or surv.estimator is None:
        print("  survival fit skipped (lifelines absent or singular fit)")
    else:
        print(f"  C-index (test) = {surv.metrics.get('c_index', float('nan')):.3f}")
        print(f"  C-index (train) = {surv.metrics.get('concordance_train', float('nan')):.3f}")

    print("training IsolationForest anomaly detector...")
    anom_model = train_anomaly_detector(ts_dataset, feature_cols)
    anom_metrics = evaluate_anomaly(anom_model, ts_dataset)
    print(f"  anomaly score vs label AUC = {anom_metrics['score_auc_vs_label']:.3f}")

    print("computing drift baseline quantiles...")
    drift_baseline = baseline_quantiles(ts_dataset, feature_cols)

    print("registering model artifacts...")
    registry_path = register(
        cal_model=cal_model,
        survival_model=surv,
        anomaly_model=anom_model,
        drift_baseline=drift_baseline,
        training_dataset=ts_dataset,
    )
    print(f"  → {registry_path}")

    print("logging to MLflow...")
    run_id = log_training_run(
        cal_model=cal_model,
        survival_model=surv,
        anomaly_metrics=anom_metrics,
        config={
            "n_features": len(feature_cols),
            "n_splits": n_splits,
            "n_total_rows": len(ts_dataset),
            "registry_version": registry_path.name,
        },
        registry_path=registry_path,
    )
    if run_id:
        print(f"  run_id = {run_id}")

    # at-risk leaderboard from the latest window
    latest = ts_dataset[ts_dataset["window_end_tick"] == ts_dataset["window_end_tick"].max()]
    leaderboard = top_at_risk(cal_model, latest, n=10)
    leaderboard.to_csv(OUT_DIR / "top_at_risk.csv", index=False)

    # Persist the latest scoring window so the dashboard doesn't have to
    # rebuild it from 16M raw samples on every reload.
    latest.to_parquet(OUT_DIR / "latest_window.parquet")

    if cal_model.shap_summary is not None:
        cal_model.shap_summary.head(15).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    pd.DataFrame([cal_model.metrics]).to_csv(OUT_DIR / "metrics.csv", index=False)

    print()
    print("top 5 features (mean |SHAP|):")
    if cal_model.shap_summary is not None:
        print(cal_model.shap_summary.head(5).to_string(index=False))
    print()
    print(f"top 10 at-risk nodes:")
    print(leaderboard.to_string(index=False))
    return cal_model.metrics


if __name__ == "__main__":
    run()
