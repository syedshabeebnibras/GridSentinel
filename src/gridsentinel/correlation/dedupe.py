"""Drop near-duplicate events before correlation.

Strategy:
  - Within each (kind, target) group, collapse events that fall within
    `window_ticks` of the previous kept event. This kills repetitive flapping
    from a single component (e.g. a GPU that throttles, recovers, throttles
    again within seconds).
  - Critical events are always kept regardless of recency.
"""
from __future__ import annotations

import pandas as pd


def dedupe(events: pd.DataFrame, window_ticks: int = 5) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    df = events.sort_values("tick").reset_index(drop=True)
    keep = [False] * len(df)
    last_seen: dict[tuple[str, str], int] = {}

    for i, row in df.iterrows():
        key = (row["kind"], row["target"])
        prev_tick = last_seen.get(key)
        if (
            prev_tick is None
            or row["tick"] - prev_tick > window_ticks
            or row["severity"] == "critical"
        ):
            keep[i] = True
            last_seen[key] = row["tick"]

    return df[keep].reset_index(drop=True)
