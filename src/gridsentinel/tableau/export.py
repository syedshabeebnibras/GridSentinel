"""Export GridSentinel data to a Tableau-friendly star schema.

Tableau Public is free but only consumes flat files (CSV/Excel/TXT) — not
parquet, and not databases. So we ship CSV everywhere with explicit
relationships documented in the README and as a sidecar JSON manifest that
matches Tableau's Data Source model.

The schema is identical to the Power BI export (one source of truth in
`gridsentinel.powerbi.export` produces both). Only the docs differ: DAX vs
Tableau Calculated Fields.
"""
from __future__ import annotations

import json
from pathlib import Path

from gridsentinel.powerbi.export import export as _export_star_schema

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "tableau"

_README = """# GridSentinel — Tableau Public dataset

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
"""

_CALC_FIELDS = """// GridSentinel — Tableau Calculated Field definitions
// Paste each block as a new Calculated Field on the indicated table.

// === fact_events / fact_incidents ===

// Alert Compression Ratio (parameterise as a calculation across the federated model)
COUNTD([fact_events].[Tick]) /  // raw event count proxy
COUNTD([fact_incidents].[Incident Id])

// Noise Rate
SUM(IF [benign] = TRUE THEN 1 ELSE 0 END) / COUNT([Tick])

// Critical Incidents
COUNTD(IF [Severity Max] = "critical" THEN [Incident Id] END)

// MTTR (hours) — duration_ticks at 12 ticks/hour
AVG([Duration Ticks]) / 12

// === fact_hourly_kpi ===

// Total Energy kWh (each row is 1 hour of fleet_kw, so kW × 1h = kWh)
SUM([Fleet Kw])

// Avg Fleet Draw kW
AVG([Fleet Kw])

// GPU Utilization %
SUM([Gpu Hours]) / (COUNT([Tick]) * [Fleet Capacity Parameter])

// === Idle waste — use a parameter for $/kWh ===

// Idle Energy Waste kWh
SUM([Idle Gpu Count]) * 0.21    // 0.3 * 700W TDP / 1000 = 0.21 kWh per idle GPU-hour

// Idle Waste USD
[Idle Energy Waste kWh] * [Electricity Price Parameter]

// === Performance per Watt ===

// Aggregate TFLOPS (H100 ~ 1000 TF FP16 at full util)
SUM([Gpu Hours]) * 1000

// Perf per Watt
[Aggregate TFLOPS] / ([Total Energy kWh] * 1000)
"""


def export(out_dir: Path = OUT_DIR) -> Path:
    """Run the shared star-schema export, then copy CSVs and write Tableau docs.

    The powerbi exporter already writes CSV alongside parquet, so we just
    symlink (or copy on Windows) the CSVs and add Tableau-specific docs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    powerbi_dir = _export_star_schema()

    for csv_path in powerbi_dir.glob("*.csv"):
        dest = out_dir / csv_path.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(csv_path)
        except OSError:
            # symlink may fail on Windows or certain mounts — fall back to copy
            dest.write_bytes(csv_path.read_bytes())

    (out_dir / "README.md").write_text(_README)
    (out_dir / "calculated_fields.txt").write_text(_CALC_FIELDS)

    # JSON manifest of the relationships — for tooling / automation
    manifest = {
        "tables": {
            "fact_events": {"file": "fact_events.csv", "grain": "raw event"},
            "fact_incidents": {"file": "fact_incidents.csv", "grain": "correlated incident"},
            "fact_hourly_kpi": {"file": "fact_hourly_kpi.csv", "grain": "fleet hour"},
            "fact_clusters": {"file": "fact_clusters.csv", "grain": "root-cause family"},
            "dim_fleet": {"file": "dim_fleet.csv", "grain": "node"},
            "dim_time": {"file": "dim_time.csv", "grain": "tick"},
        },
        "relationships": [
            {"left": "fact_events.node_id", "right": "dim_fleet.node_id", "type": "many-to-one"},
            {"left": "fact_events.tick", "right": "dim_time.tick", "type": "many-to-one"},
            {"left": "fact_hourly_kpi.tick", "right": "dim_time.tick", "type": "many-to-one"},
            {
                "left": "fact_incidents.cluster_id",
                "right": "fact_clusters.cluster_id",
                "type": "many-to-one",
            },
        ],
    }
    (out_dir / "schema.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


if __name__ == "__main__":
    path = export()
    print(f"Tableau dataset exported to {path}")
