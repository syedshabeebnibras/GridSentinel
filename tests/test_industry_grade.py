"""Tests for the v2 industry-grade PdM pipeline modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridsentinel.predict.anomaly import evaluate_anomaly, train_anomaly_detector
from gridsentinel.predict.calibrated import train_calibrated
from gridsentinel.predict.drift import baseline_quantiles, psi, psi_report
from gridsentinel.predict.ts_features import (
    build_timeseries_dataset,
    extract_window_features,
    ts_feature_columns,
)


def _make_metrics(n_nodes: int = 10, hours: int = 96, ticks_per_hour: int = 12):
    rng = np.random.default_rng(0)
    rows = []
    for h in range(hours):
        for tick_in_hour in (0, 6):
            tick = h * ticks_per_hour + tick_in_hour
            for n in range(n_nodes):
                for g in range(2):  # 2 GPUs/node for speed
                    rows.append({
                        "tick": tick,
                        "gpu_id": f"node-{n:04d}/gpu{g}",
                        "node_id": f"node-{n:04d}",
                        "rack_id": f"rack-{n // 5:03d}",
                        "zone_id": f"zone-{n // 8}",
                        "gpu_temp_c": 60 + 25 * np.sin(h * np.pi / 12) + rng.normal(0, 2),
                        "gpu_power_w": 400 + rng.normal(0, 30),
                        "gpu_sm_util": float(rng.beta(2, 3)),
                        "ecc_corrected_total": h + (5 if n == 0 and h > 40 else 0),
                        "nvlink_crc_total": h * 2 + rng.poisson(1),
                        "pcie_aer_total": h,
                    })
    return pd.DataFrame(rows)


def _make_events(n_nodes: int = 10, hours: int = 96, ticks_per_hour: int = 12):
    rows = []
    # node-0000 is the "doomed" node — gets critical at hour 80
    rows.append({
        "tick": 80 * ticks_per_hour, "kind": "ecc_uncorrectable", "scope": "gpu",
        "target": "node-0000/gpu0", "severity": "critical", "benign": False,
        "parent_event_id": None, "node_id": "node-0000",
        "rack_id": "rack-000", "zone_id": "zone-0", "feed_id": "f", "spine_id": "s",
    })
    return pd.DataFrame(rows)


def test_extract_window_features_yields_expected_columns():
    m = _make_metrics(n_nodes=5, hours=24)
    feats = extract_window_features(m)
    assert not feats.empty
    assert "node_id" in feats.columns
    # check a few signal × statistic combos
    expected = ["gpu_temp_c__mean", "gpu_temp_c__slope", "ecc_corrected_rate__max"]
    for col in expected:
        assert col in feats.columns, f"missing {col}"


def test_build_timeseries_dataset_produces_label():
    metrics = _make_metrics(n_nodes=8, hours=96)
    events = _make_events(n_nodes=8, hours=96)
    fleet = pd.DataFrame([
        {"node_id": f"node-{n:04d}", "rack_id": f"rack-{n//5:03d}",
         "zone_id": f"zone-{n//8}", "feed_id": "f", "spine_id": "s"}
        for n in range(8)
    ])
    ds = build_timeseries_dataset(metrics, events, fleet, lookback_hours=24, forecast_hours=24, step_hours=12)
    assert not ds.empty
    assert "failed_next_24h" in ds.columns
    assert {"node_id", "rack_id", "zone_id", "window_end_tick"} <= set(ds.columns)


def test_calibrated_train_runs_and_reports_brier():
    metrics = _make_metrics(n_nodes=20, hours=120)
    events = _make_events(n_nodes=20, hours=120)
    fleet = pd.DataFrame([
        {"node_id": f"node-{n:04d}", "rack_id": f"rack-{n//5:03d}",
         "zone_id": f"zone-{n//8}", "feed_id": "f", "spine_id": "s"}
        for n in range(20)
    ])
    ds = build_timeseries_dataset(metrics, events, fleet, lookback_hours=24, forecast_hours=24, step_hours=12)
    cols = ts_feature_columns(ds)
    model = train_calibrated(ds, cols, n_splits=3)
    assert "roc_auc_mean" in model.metrics
    assert "brier_mean" in model.metrics
    assert model.feature_importance is not None


def test_anomaly_detector_trains_and_scores():
    metrics = _make_metrics(n_nodes=10, hours=96)
    events = _make_events(n_nodes=10, hours=96)
    fleet = pd.DataFrame([
        {"node_id": f"node-{n:04d}", "rack_id": f"rack-{n//5:03d}",
         "zone_id": f"zone-{n//8}", "feed_id": "f", "spine_id": "s"}
        for n in range(10)
    ])
    ds = build_timeseries_dataset(metrics, events, fleet, lookback_hours=24, forecast_hours=24, step_hours=12)
    cols = ts_feature_columns(ds)
    anom = train_anomaly_detector(ds, cols, contamination=0.1)
    scores = anom.score(ds)
    assert len(scores) == len(ds)
    assert scores.min() >= 0 and scores.max() <= 1


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    edges = np.linspace(x.min(), x.max(), 11)
    assert psi(x, x, edges) < 1e-6


def test_psi_is_positive_for_shifted_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    cur = rng.normal(loc=2.0, size=1000)
    edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), 11)
    assert psi(ref, cur, edges) > 0.5  # massive shift


def test_drift_report_classifies_severity():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(size=500), "f2": rng.normal(size=500)})
    edges = baseline_quantiles(df, ["f1", "f2"])
    shifted = pd.DataFrame({"f1": rng.normal(loc=3.0, size=500), "f2": rng.normal(size=500)})
    report = psi_report(df, shifted, edges)
    assert "psi" in report.columns
    assert "severity" in report.columns
    # f1 was shifted hard → should be 'significant'; f2 not → 'stable'
    f1_sev = report.loc[report["feature"] == "f1", "severity"].iat[0]
    f2_sev = report.loc[report["feature"] == "f2", "severity"].iat[0]
    assert f1_sev in ("moderate", "significant")
    assert f2_sev == "stable"
