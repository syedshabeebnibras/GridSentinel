"""Load synthetic telemetry into tidy DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"


def load_events() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "events.parquet")


def load_utilization() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "utilization.parquet")


def load_fleet() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "fleet.parquet")


def _node_id_from_target(target: str) -> str | None:
    """Extract node id from event target (handles 'node-0042', 'node-0042/gpu3')."""
    if not isinstance(target, str):
        return None
    return target.split("/", 1)[0] if target.startswith("node-") else None


def enrich_with_topology(events: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """Add rack_id / zone_id / feed_id / spine_id columns to each event.

    Zone-scope events (e.g. cooling_failure) already carry zone in `target`, so
    we fill those directly. Node/GPU-scope events get joined via node_id.
    """
    df = events.copy()
    df["node_id"] = df["target"].map(_node_id_from_target)
    df = df.merge(fleet, on="node_id", how="left")
    # zone-scope events: their target IS the zone
    zone_mask = df["scope"] == "zone"
    df.loc[zone_mask, "zone_id"] = df.loc[zone_mask, "target"]
    return df
