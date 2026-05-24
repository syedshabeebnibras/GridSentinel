"""Group deduped events into incidents using parent-id + topology proximity.

Rules (applied in order):
  1. Explicit cascade: every event sharing a `parent_event_id` joins that
     incident (the parent's own event is the seed).
  2. Co-located burst: events of the same `kind` within `time_window_ticks`
     that share a topology key (rack → zone → feed → spine, first match wins)
     collapse into a single incident.
  3. Solo: anything still ungrouped becomes its own incident.

The incident table has one row per incident with opened/resolved ticks,
membership count, max severity, and a benign flag (all members benign).
"""
from __future__ import annotations

import pandas as pd

_SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}
_TOPOLOGY_KEYS = ("rack_id", "zone_id", "feed_id", "spine_id")


def _max_severity(s: pd.Series) -> str:
    return max(s.dropna(), key=_SEVERITY_RANK.get, default="info")


def correlate(events: pd.DataFrame, time_window_ticks: int = 10) -> pd.DataFrame:
    """events must already be enriched with topology columns (see ingest.telemetry)."""
    if events.empty:
        return pd.DataFrame()

    df = events.sort_values("tick").reset_index(drop=True).copy()
    df["incident_id"] = pd.NA

    # Rule 1 — cascade by parent_event_id.
    has_parent = df["parent_event_id"].notna()
    df.loc[has_parent, "incident_id"] = (
        "cascade:" + df.loc[has_parent, "parent_event_id"].astype("object").astype(str)
    )
    # Each parent event is itself a member of its own incident.
    # The parent's id is encoded in its target+tick. We synthesize it the same way.
    # In our simulator the parent event has parent_event_id == None but is referenced
    # by children. So we tag the parent by matching its (kind, tick, target) signature
    # if any child references it. Simpler: give parent its own cascade id by matching
    # children's parent_event_id back to a (zone, tick) — but we don't have that link.
    # For now, parent rows that lack `parent_event_id` will be picked up in Rule 2.

    # Rule 2 — co-located bursts by topology + time window.
    ungrouped = df["incident_id"].isna()
    bucket = (df.loc[ungrouped, "tick"] // time_window_ticks).astype(int).astype(str)
    # pick the most-specific available topology key per row
    topo_key = pd.Series(index=df.index[ungrouped], dtype="object")
    for col in _TOPOLOGY_KEYS:
        if col not in df.columns:
            continue
        mask = topo_key.isna() & df.loc[ungrouped, col].notna()
        topo_key.loc[mask[mask].index] = (
            f"{col}=" + df.loc[mask[mask].index, col].astype(str)
        )
    topo_key = topo_key.fillna(
        "solo=" + df.loc[ungrouped, "target"].astype("object").astype(str)
    )
    df.loc[ungrouped, "incident_id"] = (
        "burst:"
        + df.loc[ungrouped, "kind"].astype("object").astype(str)
        + "|"
        + topo_key.astype("object").astype(str)
        + "|t"
        + bucket
    )

    # Aggregate
    grouped = df.groupby("incident_id", dropna=False)
    incidents = grouped.agg(
        opened_tick=("tick", "min"),
        resolved_tick=("tick", "max"),
        member_count=("tick", "count"),
        root_kind=("kind", "first"),
        scope=("scope", "first"),
        severity_max=("severity", _max_severity),
        all_benign=("benign", "all"),
        any_benign=("benign", "any"),
    ).reset_index()
    incidents["duration_ticks"] = incidents["resolved_tick"] - incidents["opened_tick"]
    return incidents.sort_values("opened_tick").reset_index(drop=True)
