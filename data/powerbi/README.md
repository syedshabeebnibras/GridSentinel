# GridSentinel — Power BI dataset

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
