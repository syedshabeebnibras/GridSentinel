"""MLflow experiment tracking — every training run logged to mlruns/.

We log:
  - params  : feature count, lookback/horizon/step config, n_splits, model class
  - metrics : every TSCV fold metric + global headline metrics
  - artifacts: feature_importance.csv, calibration_curve.csv, metadata.json

MLflow tracking is local-file-only (file://./mlruns) — no server required.
Run `mlflow ui` from the repo root to browse experiments after training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

MLRUNS_DIR = Path(__file__).resolve().parents[3] / "mlruns"


def log_training_run(
    *,
    cal_model: Any,
    survival_model: Any | None,
    anomaly_metrics: dict[str, float],
    config: dict[str, Any],
    registry_path: Path,
    experiment: str = "gridsentinel-pdm",
) -> str | None:
    """Log a training run to MLflow. Returns the run_id, or None if mlflow
    isn't installed."""
    try:
        import mlflow
    except ImportError:
        return None

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        # params
        for k, v in config.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_class", "HistGradientBoostingClassifier+isotonic")
        mlflow.log_param("registry_path", str(registry_path))

        # headline metrics from calibrated classifier
        for k, v in cal_model.metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                continue

        # per-fold metrics → expanded as metric_fold_<i>
        for metric_name, agg in cal_model.cv_metrics.items():
            mlflow.log_metric(f"{metric_name}_mean", agg["mean"])
            mlflow.log_metric(f"{metric_name}_std", agg["std"])

        # survival
        if survival_model is not None:
            for k, v in survival_model.metrics.items():
                try:
                    mlflow.log_metric(f"surv_{k}", float(v))
                except (TypeError, ValueError):
                    continue

        # anomaly
        for k, v in anomaly_metrics.items():
            try:
                mlflow.log_metric(f"anom_{k}", float(v))
            except (TypeError, ValueError):
                continue

        # artifacts (best-effort — skip if registry didn't write them)
        for name in ("feature_importance.csv", "metadata.json", "drift_baseline.csv"):
            artifact = registry_path / name
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        return run.info.run_id
