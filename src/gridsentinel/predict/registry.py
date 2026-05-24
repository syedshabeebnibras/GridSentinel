"""Lightweight model registry — versioned artifacts under data/models/v{N}/.

Each registered version contains:
  - calibrated_model.joblib    : sklearn-pickled CalibratedFailureModel
  - survival_model.joblib       : sklearn-pickled SurvivalModel (or absent if Cox failed)
  - anomaly_model.joblib        : sklearn-pickled AnomalyModel
  - metadata.json               : feature schema, training metrics, data hash,
                                  timestamp, model versions
  - feature_importance.csv      : SHAP global summary
  - drift_baseline.csv          : reference feature distributions for PSI

SECURITY NOTE — joblib is pickle-based. These artifacts are written by this
codebase and only loaded by this codebase. They are NOT a public input format
and should never be loaded from untrusted sources. If this changes, swap to a
safer format (ONNX for the classifier, json for survival coefficients,
parquet for the anomaly model's scaler params).

This isn't MLflow Model Registry — it's a deliberately small pattern that
proves the discipline without the dependency. MLflow tracking lives in
predict/tracking.py for the experiment-comparison angle.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

REGISTRY_DIR = Path(__file__).resolve().parents[3] / "data" / "models"


def _next_version(base_dir: Path) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name[1:]) for p in base_dir.iterdir() if p.name.startswith("v") and p.name[1:].isdigit()]
    return (max(existing) if existing else 0) + 1


def _hash_dataset(df: pd.DataFrame) -> str:
    """Stable hash of (shape, column names, first/last rows) — enough to
    detect retraining-data drift without serialising the whole table."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df.head(20)).values.tobytes())
    h.update(pd.util.hash_pandas_object(df.tail(20)).values.tobytes())
    return h.hexdigest()[:16]


def register(
    cal_model: Any,
    survival_model: Any | None,
    anomaly_model: Any,
    drift_baseline: pd.DataFrame,
    training_dataset: pd.DataFrame,
    base_dir: Path = REGISTRY_DIR,
) -> Path:
    version = _next_version(base_dir)
    out = base_dir / f"v{version}"
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(cal_model, out / "calibrated_model.joblib")
    joblib.dump(anomaly_model, out / "anomaly_model.joblib")
    if survival_model is not None:
        joblib.dump(survival_model, out / "survival_model.joblib")

    if cal_model.feature_importance is not None:
        cal_model.feature_importance.to_csv(out / "feature_importance.csv", index=False)

    drift_baseline.to_csv(out / "drift_baseline.csv", index=False)

    metadata: dict[str, Any] = {
        "version": version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_hash": _hash_dataset(training_dataset),
        "n_training_rows": int(len(training_dataset)),
        "n_features": int(len(cal_model.feature_names)),
        "feature_names": list(cal_model.feature_names),
        "label": "failed_next_24h",
        "metrics": cal_model.metrics,
        "cv_metrics": cal_model.cv_metrics,
        "survival_metrics": (
            survival_model.metrics if (survival_model is not None) else None
        ),
        "anomaly_contamination": getattr(anomaly_model, "contamination", None),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (base_dir / "LATEST").write_text(f"v{version}\n")
    return out


def load_latest(base_dir: Path = REGISTRY_DIR) -> dict[str, Any] | None:
    latest_file = base_dir / "LATEST"
    if not latest_file.exists():
        return None
    version_dir = base_dir / latest_file.read_text().strip()
    if not version_dir.exists():
        return None
    return {
        "version": version_dir.name,
        "metadata": json.loads((version_dir / "metadata.json").read_text()),
        "calibrated": joblib.load(version_dir / "calibrated_model.joblib"),
        "anomaly": joblib.load(version_dir / "anomaly_model.joblib"),
        "survival": (
            joblib.load(version_dir / "survival_model.joblib")
            if (version_dir / "survival_model.joblib").exists() else None
        ),
        "drift_baseline": pd.read_csv(version_dir / "drift_baseline.csv"),
    }
