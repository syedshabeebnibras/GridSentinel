# GridSentinel — Tableau Public dataset

Same star schema as the Power BI export, formatted for Tableau Public (which
does not read parquet — CSV only).

## Tables

| File | Grain | Joins |
|---|---|---|
| `fact_events.csv` | one row per raw telemetry event | `node_id → dim_fleet`, `tick → dim_time` |
| `fact_incidents.csv` | one row per correlated incident | `cluster_id → fact_clusters` |
| `fact_hourly_kpi.csv` | one row per hour | `tick → dim_time` |
| `fact_clusters.csv` | one row per root-cause family | — |
| `dim_fleet.csv` | one row per node (rack/zone/feed/spine) | — |
| `dim_time.csv` | one row per tick (datetime/hour/weekday) | — |

## How to load in Tableau Public

1. Open Tableau Public Desktop (free download — Win/Mac).
2. **Connect → Text file** → pick `fact_events.csv`.
3. In the Data Source pane, drag in the other CSV files to create a federated
   model. Use these relationships:
   - `fact_events.node_id  =  dim_fleet.node_id`     (many-to-one)
   - `fact_events.tick     =  dim_time.tick`         (many-to-one)
   - `fact_hourly_kpi.tick =  dim_time.tick`         (many-to-one)
   - `fact_incidents.cluster_id = fact_clusters.cluster_id`  (many-to-one)
4. Open `calculated_fields.txt` and paste each formula in as a new
   Calculated Field on the matching table (table name appears as a header).

## Suggested worksheets

- **KPI cards**: Alert Compression, Critical Incidents, MTTR, Perf/Watt, Idle Waste $.
- **Line chart**: `dim_time.datetime` on Columns, `Avg Fleet Draw kW` + `GPU Utilization %` dual-axis on Rows.
- **Treemap**: `fact_clusters.cluster_label` size = `count`.
- **Heatmap**: rows = `dim_fleet.zone_id`, columns = `dim_time.hour`, color = critical-incident count.
- **Highlight table** for top-10 at-risk nodes (after running `python -m gridsentinel.predict.score`).

## Refreshing

After re-running the simulator, run `python -m gridsentinel.tableau.export` and
in Tableau: **Data → Refresh All Extracts**.
