import pandas as pd

from gridsentinel.correlation.correlate import correlate
from gridsentinel.correlation.dedupe import dedupe


def _make_events(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_dedupe_collapses_flapping():
    # same (kind, target) flapping every tick — only the first within window kept
    events = _make_events(
        [
            {"tick": 0, "kind": "thermal_throttle", "target": "g1", "severity": "info"},
            {"tick": 1, "kind": "thermal_throttle", "target": "g1", "severity": "info"},
            {"tick": 2, "kind": "thermal_throttle", "target": "g1", "severity": "info"},
            {"tick": 50, "kind": "thermal_throttle", "target": "g1", "severity": "info"},
        ]
    )
    out = dedupe(events, window_ticks=5)
    # ticks 0 and 50 survive; 1 and 2 collapsed
    assert list(out["tick"]) == [0, 50]


def test_dedupe_always_keeps_critical():
    events = _make_events(
        [
            {"tick": 0, "kind": "psu_trip", "target": "rack-1", "severity": "info"},
            {"tick": 1, "kind": "psu_trip", "target": "rack-1", "severity": "critical"},
        ]
    )
    out = dedupe(events, window_ticks=10)
    assert len(out) == 2  # critical kept even within window


def test_correlate_cascades_share_parent():
    events = _make_events(
        [
            {
                "tick": 100,
                "kind": "thermal_throttle",
                "scope": "node",
                "target": "node-0001",
                "severity": "warn",
                "benign": False,
                "parent_event_id": "cooling-zone-0-100",
                "rack_id": "rack-000",
                "zone_id": "zone-0",
                "feed_id": "zone-0/feed-0",
                "spine_id": "spine-0",
            },
            {
                "tick": 100,
                "kind": "thermal_throttle",
                "scope": "node",
                "target": "node-0002",
                "severity": "warn",
                "benign": False,
                "parent_event_id": "cooling-zone-0-100",
                "rack_id": "rack-000",
                "zone_id": "zone-0",
                "feed_id": "zone-0/feed-0",
                "spine_id": "spine-0",
            },
        ]
    )
    inc = correlate(events)
    assert len(inc) == 1
    assert inc.iloc[0]["member_count"] == 2


def test_correlate_groups_by_rack_within_window():
    rows = []
    for i in range(5):
        rows.append(
            {
                "tick": i,
                "kind": "thermal_throttle",
                "scope": "gpu",
                "target": f"node-000{i}/gpu0",
                "severity": "info",
                "benign": True,
                "parent_event_id": None,
                "rack_id": "rack-000",
                "zone_id": "zone-0",
                "feed_id": "zone-0/feed-0",
                "spine_id": "spine-0",
            }
        )
    inc = correlate(_make_events(rows), time_window_ticks=10)
    # all 5 events same rack, same window → 1 incident
    assert len(inc) == 1
    assert inc.iloc[0]["member_count"] == 5
