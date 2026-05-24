import pandas as pd

from gridsentinel.rootcause.cluster import cluster_incidents, top_recurring
from gridsentinel.rootcause.features import featurize


def _toy_incidents(n: int = 60) -> pd.DataFrame:
    rows = []
    # 30 morning thermal bursts on gpu
    for i in range(30):
        rows.append({
            "incident_id": f"a{i}",
            "root_kind": "thermal_throttle",
            "scope": "gpu",
            "member_count": 10,
            "severity_max": "info",
            "all_benign": False,
            "any_benign": False,
            "opened_tick": 9 * 12 + i,        # ~9am
            "resolved_tick": 9 * 12 + i + 2,
            "duration_ticks": 2,
        })
    # 30 afternoon critical cooling cascades
    for i in range(30):
        rows.append({
            "incident_id": f"b{i}",
            "root_kind": "thermal_throttle",
            "scope": "node",
            "member_count": 100,
            "severity_max": "critical",
            "all_benign": False,
            "any_benign": False,
            "opened_tick": 15 * 12 + i,       # ~3pm
            "resolved_tick": 15 * 12 + i + 1,
            "duration_ticks": 1,
        })
    return pd.DataFrame(rows)


def test_featurize_shapes_and_cols():
    inc = _toy_incidents()
    f = featurize(inc)
    assert len(f) == len(inc)
    # one-hot columns present
    assert "kind_thermal_throttle" in f.columns
    assert "scope_gpu" in f.columns
    assert "hod_sin" in f.columns
    assert "hod_cos" in f.columns
    # one-hot is binary
    assert set(f["kind_thermal_throttle"].unique()) <= {0.0, 1.0}


def test_clustering_separates_morning_from_afternoon():
    inc = _toy_incidents()
    f = featurize(inc)
    clustered = cluster_incidents(f, inc, min_cluster_size=5)
    # the two synthetic groups should land in distinct clusters
    morning = clustered.iloc[:30]["cluster_id"]
    afternoon = clustered.iloc[30:]["cluster_id"]
    real_morning = set(morning) - {-1}
    real_afternoon = set(afternoon) - {-1}
    assert real_morning, "morning group should produce at least one cluster"
    assert real_afternoon, "afternoon group should produce at least one cluster"
    assert real_morning.isdisjoint(real_afternoon)


def test_top_recurring_returns_dataframe():
    inc = _toy_incidents()
    f = featurize(inc)
    clustered = cluster_incidents(f, inc, min_cluster_size=5)
    out = top_recurring(clustered, n=5)
    assert {"cluster_id", "cluster_label", "count"} <= set(out.columns)
    assert len(out) >= 1
