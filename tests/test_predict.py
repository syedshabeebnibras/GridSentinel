import numpy as np
import pandas as pd

from gridsentinel.predict.features import X_y, build_training_set, feature_columns
from gridsentinel.predict.model import time_based_split, top_at_risk, train


def _toy_enriched(n_nodes: int = 30, hours: int = 96, ticks_per_hour: int = 12):
    rng = np.random.default_rng(0)
    nodes = [f"node-{i:04d}" for i in range(n_nodes)]
    rack = {n: f"rack-{i//5:03d}" for i, n in enumerate(nodes)}
    zone = {n: f"zone-{i//15}" for i, n in enumerate(nodes)}
    feed = {n: f"feed-{i//20}" for i, n in enumerate(nodes)}
    spine = {n: f"spine-{i % 2}" for i, n in enumerate(nodes)}

    rows = []
    for h in range(hours):
        tick = h * ticks_per_hour
        # 10% of nodes get random "warn" events per hour
        for n in rng.choice(nodes, size=max(1, n_nodes // 10), replace=False):
            rows.append({
                "tick": tick, "kind": "thermal_throttle", "scope": "gpu",
                "target": f"{n}/gpu0", "severity": "warn", "benign": False,
                "parent_event_id": None, "node_id": n,
                "rack_id": rack[n], "zone_id": zone[n],
                "feed_id": feed[n], "spine_id": spine[n],
            })
        # nodes with prior warns get a critical with high probability ~6h later
        if h > 6:
            recent = [r for r in rows if r["tick"] == (h - 6) * ticks_per_hour]
            for r in recent[:2]:  # 2 of them fail
                rows.append({
                    "tick": tick, "kind": "psu_trip", "scope": "node",
                    "target": r["node_id"], "severity": "critical", "benign": False,
                    "parent_event_id": None, "node_id": r["node_id"],
                    "rack_id": r["rack_id"], "zone_id": r["zone_id"],
                    "feed_id": r["feed_id"], "spine_id": r["spine_id"],
                })
    return pd.DataFrame(rows), nodes, rack, zone, feed, spine


def test_features_build_with_expected_columns():
    enriched, nodes, rack, zone, feed, spine = _toy_enriched(n_nodes=30, hours=72)
    fleet = pd.DataFrame([
        {"node_id": n, "rack_id": rack[n], "zone_id": zone[n], "feed_id": feed[n], "spine_id": spine[n]}
        for n in nodes
    ])
    util = pd.DataFrame([
        {"tick": h * 12, "gpu_id": f"{n}/gpu0", "util": 0.5}
        for h in range(72) for n in nodes
    ])
    dataset = build_training_set(enriched, util, fleet, lookback_hours=12, forecast_hours=12, step_hours=6)
    assert not dataset.empty
    assert "failed_next_24h" in dataset.columns
    # all feature columns should be present
    for col in feature_columns():
        assert col in dataset.columns, f"missing feature column {col}"


def test_time_split_doesnt_overlap():
    enriched, nodes, rack, zone, feed, spine = _toy_enriched(n_nodes=20, hours=72)
    fleet = pd.DataFrame([
        {"node_id": n, "rack_id": rack[n], "zone_id": zone[n], "feed_id": feed[n], "spine_id": spine[n]}
        for n in nodes
    ])
    util = pd.DataFrame([
        {"tick": h * 12, "gpu_id": f"{n}/gpu0", "util": 0.5}
        for h in range(72) for n in nodes
    ])
    dataset = build_training_set(enriched, util, fleet, lookback_hours=12, forecast_hours=12, step_hours=6)
    train_df, test_df = time_based_split(dataset, test_fraction=0.3)
    assert train_df["window_end_tick"].max() <= test_df["window_end_tick"].min()


def test_train_runs_and_returns_metrics():
    enriched, nodes, rack, zone, feed, spine = _toy_enriched(n_nodes=30, hours=96)
    fleet = pd.DataFrame([
        {"node_id": n, "rack_id": rack[n], "zone_id": zone[n], "feed_id": feed[n], "spine_id": spine[n]}
        for n in nodes
    ])
    util = pd.DataFrame([
        {"tick": h * 12, "gpu_id": f"{n}/gpu0", "util": 0.5}
        for h in range(96) for n in nodes
    ])
    dataset = build_training_set(enriched, util, fleet, lookback_hours=12, forecast_hours=12, step_hours=6)
    model = train(dataset)
    assert "roc_auc" in model.metrics
    # synthetic signal should be learnable (warn → critical 6h later)
    assert model.metrics["roc_auc"] >= 0.55 or np.isnan(model.metrics["roc_auc"])
    assert model.feature_importance is not None
    assert len(model.feature_importance) == len(feature_columns())


def test_top_at_risk_returns_top_n_sorted():
    enriched, nodes, rack, zone, feed, spine = _toy_enriched(n_nodes=20, hours=96)
    fleet = pd.DataFrame([
        {"node_id": n, "rack_id": rack[n], "zone_id": zone[n], "feed_id": feed[n], "spine_id": spine[n]}
        for n in nodes
    ])
    util = pd.DataFrame([
        {"tick": h * 12, "gpu_id": f"{n}/gpu0", "util": 0.5}
        for h in range(96) for n in nodes
    ])
    dataset = build_training_set(enriched, util, fleet, lookback_hours=12, forecast_hours=12, step_hours=6)
    model = train(dataset)
    latest = dataset[dataset["window_end_tick"] == dataset["window_end_tick"].max()]
    out = top_at_risk(model, latest, n=5)
    assert len(out) <= 5
    if len(out) > 1:
        assert (out["failure_risk_24h"].diff().dropna() <= 0).all()
