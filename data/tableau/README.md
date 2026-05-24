# GridSentinel — Tableau dataset

Star-schema CSV export plus a packaged Tableau workbook ready to open in
Tableau Public Desktop (free, macOS / Windows).

## What's in this folder

| File | Purpose |
|---|---|
| `GridSentinel.twbx` | **Packaged Tableau workbook** — bundles all 6 CSVs + workbook XML. Double-click to open. |
| `*.csv` | Star-schema flat files: `fact_events`, `fact_incidents`, `fact_hourly_kpi`, `fact_clusters`, `dim_fleet`, `dim_time`. |
| `calculated_fields.txt` | Reference list of Tableau Calculated Fields (compression, MTTR, perf/Watt, etc). |
| `schema.json` | Machine-readable relationship manifest. |

## How to use

### 1. Open the workbook

```bash
# macOS
open GridSentinel.twbx
```

Tableau Public Desktop opens and 6 data sources appear in the left pane. The
relationships (events↔fleet via `node_id`, kpi↔time via `tick`, etc.) are
defined in `schema.json` and need to be wired in Tableau's Data Model view on
first open.

### 2. Build visuals

Drag fields to the Rows / Columns shelves to build sheets. Suggested starts:

- **KPI cards** — Alert Compression, Critical Incidents, MTTR (hours), Perf/Watt
- **Bar chart** — `fact_incidents[root_kind]` on Columns, `COUNTD(incident_id)` on Rows
- **Heatmap** — Rows = `dim_fleet[zone_id]`, Columns = `dim_time[hour]`, Color = critical-incident count
- **Treemap** — `fact_clusters[cluster_label]` sized by `count`

### 3. Publish to Tableau Public

`File → Save to Tableau Public As...` — public URL you can link from your
portfolio / resume.

## Regenerating

```bash
python -m gridsentinel.tableau.export    # regenerate CSVs from latest sim
python -m gridsentinel.bi.twbx           # rebuild GridSentinel.twbx
```
