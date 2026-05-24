"""Export GridSentinel data to a Power BI-friendly star schema.

Layout (parquet + CSV — Power BI Desktop handles both natively):

  data/powerbi/
    fact_events.parquet         # one row per raw event, enriched with topology
    fact_incidents.parquet      # one row per correlated incident
    fact_hourly_kpi.parquet     # one row per hour: util, power_kw, gpu_hours
    fact_clusters.parquet       # one row per cluster with summary stats
    dim_fleet.parquet           # node-level topology dimension
    dim_time.parquet            # date/hour/weekday dimension for the sim window
    README.md                   # how to load in Power BI Desktop
    measures.dax                # paste-ready DAX measures for the model

Each fact references dim_fleet by node_id and dim_time by tick. The DAX file
gives you the headline measures (Availability, MTTR, Compression Ratio,
Perf/Watt, Idle Waste $) without re-deriving them.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from gridsentinel.correlation.correlate import correlate
from gridsentinel.correlation.dedupe import dedupe
from gridsentinel.ingest.telemetry import (
    enrich_with_topology,
    load_events,
    load_fleet,
    load_utilization,
)
from gridsentinel.kpi.efficiency import GPU_PEAK_TFLOPS_FP16, power_timeseries
from gridsentinel.rootcause.cluster import cluster_incidents, top_recurring
from gridsentinel.rootcause.features import featurize

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "powerbi"

_DAX_MEASURES = """// GridSentinel — paste-ready DAX measures
// Load all parquet files in data/powerbi/ as tables, then add these measures.

// === SLA ===
Total Events =
COUNTROWS ( fact_events )

Total Incidents =
COUNTROWS ( fact_incidents )

Alert Compression Ratio =
DIVIDE ( [Total Events], [Total Incidents], BLANK () )

Noise Rate =
DIVIDE (
    CALCULATE ( COUNTROWS ( fact_events ), fact_events[benign] = TRUE () ),
    [Total Events]
)

Critical Incidents =
CALCULATE ( COUNTROWS ( fact_incidents ), fact_incidents[severity_max] = "critical" )

MTTR (hours) =
AVERAGEX ( fact_incidents, fact_incidents[duration_ticks] / 12 )

// === Energy / efficiency ===
Total Energy kWh =
SUM ( fact_hourly_kpi[fleet_kw] )

Aggregate TFLOPS =
SUMX ( fact_hourly_kpi, fact_hourly_kpi[gpu_hours] * 1000 )  // H100 ~ 1000 TF FP16

Perf per Watt (TFLOPS/W) =
DIVIDE ( [Aggregate TFLOPS], [Total Energy kWh] * 1000 )

Avg Fleet Draw kW =
AVERAGE ( fact_hourly_kpi[fleet_kw] )

Idle GPU-hours =
CALCULATE (
    SUMX ( fact_hourly_kpi, fact_hourly_kpi[idle_gpu_count] ),
    fact_hourly_kpi[idle_gpu_count] > 0
)

Idle Energy Waste kWh =
[Idle GPU-hours] * 0.21  // 0.3 * 700W / 1000 = 0.21 kWh per idle GPU-hour

Idle Waste $ =
[Idle Energy Waste kWh] * 0.06  // $/kWh — adjust to PPA rate

// === Capacity ===
Fleet Capacity GPU-h-per-hour =
COUNTROWS ( dim_fleet ) * 8

GPU Utilization % =
DIVIDE ( SUM ( fact_hourly_kpi[gpu_hours] ), [Fleet Capacity GPU-h-per-hour] * COUNTROWS ( fact_hourly_kpi ) )

Headroom % =
1 - [GPU Utilization %]
"""

_README = """# GridSentinel — Power BI dataset

This folder contains a Power BI-ready star schema exported from the GridSentinel
simulator. Each file is a parquet table that Power BI Desktop can load
directly (`Get Data → Parquet`).

## Tables

| Table | Grain | Joins |
|---|---|---|
| `fact_events` | one row per raw telemetry event | `node_id → dim_fleet`, `tick → dim_time` |
| `fact_incidents` | one row per correlated incident | `cluster_id → fact_clusters` |
| `fact_hourly_kpi` | one row per hour | `tick → dim_time` |
| `fact_clusters` | one row per root-cause family | — |
| `dim_fleet` | one row per node (rack/zone/feed/spine) | — |
| `dim_time` | one row per tick (datetime/hour/weekday) | — |

## How to load

1. Open Power BI Desktop.
2. `Home → Get Data → More → Parquet → Connect`.
3. Pick each `.parquet` file in this folder. (Or use `Folder` connector to load all at once.)
4. In the Model view, create relationships:
   - `fact_events[node_id] → dim_fleet[node_id]` (many-to-one)
   - `fact_events[tick] → dim_time[tick]` (many-to-one)
   - `fact_hourly_kpi[tick] → dim_time[tick]` (many-to-one)
   - `fact_incidents[cluster_id] → fact_clusters[cluster_id]` (many-to-one)
5. Open `measures.dax`, paste each measure into a new measure on the matching
   table (table name appears as a `// === ... ===` header comment).

## Suggested visuals

- **KPI cards** — Alert Compression Ratio, Critical Incidents, MTTR, Perf/Watt, Idle Waste $.
- **Line chart** — `dim_time[hour]` on axis, `Avg Fleet Draw kW` and `GPU Utilization %` as values.
- **Treemap or bar** — `fact_clusters[cluster_label]` by `count` (top recurring root causes).
- **Map / matrix** — `dim_fleet[zone_id]` by incident count (heatmap of bad zones).
- **Decomposition tree** — drill from Critical Incidents → zone → rack → node → kind.

## Refreshing

Re-run `python -m gridsentinel.powerbi.export` after any new simulation, then
`Home → Refresh` in Power BI Desktop.
"""


def _build_dim_time(max_tick: int, ticks_per_hour: int = 12) -> pd.DataFrame:
    start = datetime(2026, 5, 1)
    rows = []
    for tick in range(max_tick + 1):
        ts = start + timedelta(hours=tick / ticks_per_hour)
        rows.append(
            {
                "tick": tick,
                "datetime": ts,
                "date": ts.date(),
                "hour": ts.hour,
                "day_of_week": ts.strftime("%A"),
                "is_weekend": ts.weekday() >= 5,
            }
        )
    return pd.DataFrame(rows)


def export(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events()
    fleet = load_fleet()
    util = load_utilization()

    enriched = enrich_with_topology(events, fleet)
    deduped = dedupe(enriched, window_ticks=5)
    incidents = correlate(deduped, time_window_ticks=15)
    feats = featurize(incidents)
    clustered = cluster_incidents(feats, incidents, min_cluster_size=8)
    clusters = top_recurring(clustered, n=50)

    # fact_hourly_kpi
    ts = power_timeseries(util, gpus_per_node=8)
    idle_per_tick = (
        util.assign(is_idle=lambda d: d["util"] < 0.05)
        .groupby("tick")["is_idle"]
        .sum()
        .rename("idle_gpu_count")
        .reset_index()
    )
    hourly_kpi = ts.merge(idle_per_tick, on="tick", how="left").rename(
        columns={"gpu_hours_in_tick": "gpu_hours"}
    )
    hourly_kpi["idle_gpu_count"] = hourly_kpi["idle_gpu_count"].fillna(0).astype(int)

    # dim_time
    max_tick = int(max(events["tick"].max(), util["tick"].max()))
    dim_time = _build_dim_time(max_tick)

    # write
    enriched.to_parquet(out_dir / "fact_events.parquet")
    clustered.to_parquet(out_dir / "fact_incidents.parquet")
    hourly_kpi.to_parquet(out_dir / "fact_hourly_kpi.parquet")
    clusters.to_parquet(out_dir / "fact_clusters.parquet")
    fleet.to_parquet(out_dir / "dim_fleet.parquet")
    dim_time.to_parquet(out_dir / "dim_time.parquet")

    # CSV duplicates for cross-tool friendliness (Tableau / Excel / Sheets)
    enriched.to_csv(out_dir / "fact_events.csv", index=False)
    clustered.to_csv(out_dir / "fact_incidents.csv", index=False)
    hourly_kpi.to_csv(out_dir / "fact_hourly_kpi.csv", index=False)
    clusters.to_csv(out_dir / "fact_clusters.csv", index=False)
    fleet.to_csv(out_dir / "dim_fleet.csv", index=False)
    dim_time.to_csv(out_dir / "dim_time.csv", index=False)

    (out_dir / "measures.dax").write_text(_DAX_MEASURES)
    (out_dir / "README.md").write_text(_README)

    return out_dir


if __name__ == "__main__":
    path = export()
    print(f"Power BI dataset exported to {path}")
