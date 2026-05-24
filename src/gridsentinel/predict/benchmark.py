"""Benchmark: compare the deep TCN baseline against the calibrated GBM.

Runs both on the same data and same TSCV folds, then prints a side-by-side
comparison table.

Honest framing: gradient boosting on engineered features tends to win on
tabular telemetry. Showing this empirically is part of the rigor — we
chose GBM because it benchmarks well, not because we didn't try DL.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridsentinel.ingest.telemetry import enrich_with_topology, load_events, load_fleet
from gridsentinel.predict.calibrated import train_calibrated
from gridsentinel.predict.deep_baseline import train_tcn
from gridsentinel.predict.ts_features import build_timeseries_dataset, ts_feature_columns
from gridsentinel.predict.score import load_metrics

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "predict"


def run(epochs: int = 8, n_splits: int = 3) -> dict:
    events = load_events()
    fleet = load_fleet()
    metrics = load_metrics()
    enriched = enrich_with_topology(events, fleet)

    print(f"benchmarking on {len(metrics):,} metric samples...")

    print("\n[1/2] training GBM (calibrated, TSCV)...")
    ts_dataset = build_timeseries_dataset(metrics, enriched, fleet)
    feature_cols = ts_feature_columns(ts_dataset)
    gbm = train_calibrated(ts_dataset, feature_cols, n_splits=n_splits)
    print(f"  ROC AUC = {gbm.metrics['roc_auc_mean']:.3f} ± {gbm.metrics['roc_auc_std']:.3f}")

    print(f"\n[2/2] training TCN ({epochs} epochs/fold)...")
    tcn = train_tcn(metrics, enriched, fleet, n_splits=n_splits, epochs=epochs)
    if "error" in tcn.metrics:
        print(f"  skipped: {tcn.metrics['error']}")
        return {"gbm": gbm.metrics, "tcn": tcn.metrics}
    print(f"  device  = {tcn.device}, params = {tcn.n_params:,}")
    print(f"  ROC AUC = {tcn.metrics['roc_auc_mean']:.3f} ± {tcn.metrics['roc_auc_std']:.3f}")

    rows = []
    for label, m in [("GBM (calibrated)", gbm.metrics), ("TCN (deep)", tcn.metrics)]:
        rows.append({
            "model": label,
            "roc_auc_mean": m.get("roc_auc_mean", float("nan")),
            "roc_auc_std": m.get("roc_auc_std", float("nan")),
            "pr_auc_mean": m.get("pr_auc_mean", float("nan")),
            "brier_mean": m.get("brier_mean", float("nan")),
            "precision_at_10_mean": m.get("precision_at_10_mean", float("nan")),
            "lift_at_10_mean": m.get("lift_at_10_mean", float("nan")),
        })
    comparison = pd.DataFrame(rows)
    print("\n=== side-by-side ===")
    print(comparison.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_DIR / "benchmark_gbm_vs_tcn.csv", index=False)
    (OUT_DIR / "benchmark_gbm_vs_tcn.json").write_text(
        json.dumps(
            {
                "gbm": gbm.metrics, "tcn": tcn.metrics,
                "tcn_n_params": tcn.n_params, "tcn_device": tcn.device,
            },
            indent=2, default=str,
        )
    )

    winner = "GBM" if (
        gbm.metrics.get("roc_auc_mean", 0) >= tcn.metrics.get("roc_auc_mean", 0)
    ) else "TCN"
    print(f"\nwinner by ROC AUC: {winner}")
    return {"gbm": gbm.metrics, "tcn": tcn.metrics, "winner": winner}


if __name__ == "__main__":
    run()
