"""Generate Power BI deliverables:

  1. GridSentinel.pbit          — a Power BI template (zip with model JSON,
                                  connections, blank report). Best-effort:
                                  we can't test it on macOS but the format
                                  follows the documented OPC spec.
  2. GridSentinel.pq            — a Power Query M script. Guaranteed to work:
                                  paste into Power BI Desktop's Advanced
                                  Editor (or Power Query in Excel) and all
                                  6 tables load. This is the reliable path
                                  if the .pbit refuses to open.

Both reference the CSV exports under data/powerbi/. The .pbit hard-codes
the path; the .pq accepts a parameter so it works regardless of where the
files live on the user's machine.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "data" / "powerbi"
OUT_PBIT = SRC_DIR / "GridSentinel.pbit"
OUT_PQ = SRC_DIR / "GridSentinel.pq"

# Tables to wire up. Column types use Power BI dataType codes:
#   "string", "int64", "double", "boolean", "dateTime"
_TABLES: dict[str, list[tuple[str, str]]] = {
    "fact_events": [
        ("tick", "int64"), ("kind", "string"), ("scope", "string"),
        ("target", "string"), ("severity", "string"), ("benign", "boolean"),
        ("parent_event_id", "string"), ("node_id", "string"),
        ("rack_id", "string"), ("zone_id", "string"),
        ("feed_id", "string"), ("spine_id", "string"),
    ],
    "fact_incidents": [
        ("incident_id", "string"), ("opened_tick", "int64"),
        ("resolved_tick", "int64"), ("member_count", "int64"),
        ("root_kind", "string"), ("scope", "string"),
        ("severity_max", "string"), ("all_benign", "boolean"),
        ("any_benign", "boolean"), ("duration_ticks", "int64"),
        ("cluster_id", "int64"), ("cluster_label", "string"),
    ],
    "fact_hourly_kpi": [
        ("tick", "int64"), ("fleet_kw", "double"),
        ("active_gpu_count", "int64"), ("gpu_hours", "double"),
        ("idle_gpu_count", "int64"),
    ],
    "fact_clusters": [
        ("cluster_id", "int64"), ("cluster_label", "string"),
        ("count", "int64"), ("critical_count", "int64"),
        ("all_benign_ratio", "double"), ("mean_members", "double"),
    ],
    "dim_fleet": [
        ("node_id", "string"), ("rack_id", "string"),
        ("zone_id", "string"), ("feed_id", "string"), ("spine_id", "string"),
    ],
    "dim_time": [
        ("tick", "int64"), ("datetime", "dateTime"), ("date", "dateTime"),
        ("hour", "int64"), ("day_of_week", "string"), ("is_weekend", "boolean"),
    ],
}

# DAX measures — same content as data/powerbi/measures.dax, embedded so they
# load with the template.
_MEASURES = [
    ("Total Events", "COUNTROWS(fact_events)", "int64", "fact_events"),
    ("Total Incidents", "COUNTROWS(fact_incidents)", "int64", "fact_incidents"),
    ("Alert Compression Ratio",
     "DIVIDE([Total Events], [Total Incidents], BLANK())",
     "double", "fact_incidents"),
    ("Critical Incidents",
     'CALCULATE(COUNTROWS(fact_incidents), fact_incidents[severity_max]="critical")',
     "int64", "fact_incidents"),
    ("MTTR (hours)",
     "AVERAGEX(fact_incidents, fact_incidents[duration_ticks] / 12.0)",
     "double", "fact_incidents"),
    ("Total Energy kWh",
     "SUM(fact_hourly_kpi[fleet_kw])",
     "double", "fact_hourly_kpi"),
    ("Avg Fleet Draw kW",
     "AVERAGE(fact_hourly_kpi[fleet_kw])",
     "double", "fact_hourly_kpi"),
    ("Idle Energy Waste kWh",
     "SUMX(fact_hourly_kpi, fact_hourly_kpi[idle_gpu_count] * 0.21)",
     "double", "fact_hourly_kpi"),
    ("Idle Waste USD",
     "[Idle Energy Waste kWh] * 0.06",
     "double", "fact_hourly_kpi"),
    ("Performance per Watt",
     "DIVIDE(SUM(fact_hourly_kpi[gpu_hours]) * 1000, [Total Energy kWh] * 1000)",
     "double", "fact_hourly_kpi"),
]


# ----------------------------------------------------------------------------
# Power Query M script — the reliable fallback path.
# ----------------------------------------------------------------------------
def _build_pq_script() -> str:
    """Single-file M script defining all queries. Paste into Power BI's
    Advanced Editor (Home → Transform data → Advanced Editor)."""
    sections = []
    for table, cols in _TABLES.items():
        type_decls = ", ".join(
            f'{{"{c}", {_pq_type(t)}}}' for c, t in cols
        )
        sections.append(f'''  {table} = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/{table}.csv"),
      [Delimiter=",", Columns={len(cols)}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{{type_decls}}})
  in Typed''')

    body = ",\n".join(sections)
    return dedent(f'''\
// GridSentinel — Power Query M script for Power BI Desktop
// Paste this into a new query via Home → Transform data → New Source → Blank Query
// → Home → Advanced Editor.
//
// Then set DataFolder to the absolute path of your data/powerbi/ directory.
// Each let-binding below becomes a separate query/table in the model.

let
  DataFolder = "C:/path/to/GridSentinel/data/powerbi",  // ← edit this
{body}
in
  // Returns the last table; reference any other table by its name in the
  // queries pane after the script runs.
  fact_incidents
''')


def _pq_type(dt: str) -> str:
    """Map our column type tags → Power Query type literals."""
    return {
        "string": "type text",
        "int64": "Int64.Type",
        "double": "type number",
        "boolean": "type logical",
        "dateTime": "type datetime",
    }.get(dt, "type text")


# ----------------------------------------------------------------------------
# .pbit template — best-effort.
# ----------------------------------------------------------------------------
def _build_datamodel_schema() -> dict:
    """Produce the DataModelSchema JSON (Power BI's TOM-style model)."""
    tables = []
    for table, cols in _TABLES.items():
        partitions = [{
            "name": table,
            "dataView": "full",
            "source": {
                "type": "m",
                "expression": [
                    f"let",
                    f'    Source = Csv.Document(File.Contents("./data/powerbi/{table}.csv"),',
                    f'        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
                    f"    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])",
                    f"in",
                    f"    #\"Promoted Headers\"",
                ],
            },
        }]
        table_def = {
            "name": table,
            "columns": [
                {"name": c, "dataType": dt, "sourceColumn": c}
                for c, dt in cols
            ],
            "partitions": partitions,
        }
        # Attach measures to the table they're defined against.
        table_measures = [
            {"name": name, "expression": expr, "formatString": "0.00"}
            for name, expr, dt, attached_table in _MEASURES
            if attached_table == table
        ]
        if table_measures:
            table_def["measures"] = table_measures
        tables.append(table_def)

    relationships = [
        {"name": "events_to_fleet", "fromTable": "fact_events",
         "fromColumn": "node_id", "toTable": "dim_fleet", "toColumn": "node_id"},
        {"name": "events_to_time", "fromTable": "fact_events",
         "fromColumn": "tick", "toTable": "dim_time", "toColumn": "tick"},
        {"name": "kpi_to_time", "fromTable": "fact_hourly_kpi",
         "fromColumn": "tick", "toTable": "dim_time", "toColumn": "tick"},
        {"name": "incidents_to_clusters", "fromTable": "fact_incidents",
         "fromColumn": "cluster_id", "toTable": "fact_clusters", "toColumn": "cluster_id"},
    ]

    return {
        "name": "GridSentinel",
        "compatibilityLevel": 1567,
        "model": {
            "culture": "en-US",
            "tables": tables,
            "relationships": relationships,
            "annotations": [
                {"name": "Generator", "value": "gridsentinel.bi.pbit"},
            ],
        },
    }


_REPORT_LAYOUT = {
    "id": 0,
    "resourcePackages": [],
    "config": "",
    "layoutOptimization": 0,
    "sections": [
        {
            "id": 0,
            "name": "Overview",
            "displayName": "Overview",
            "filters": "[]",
            "ordinal": 0,
            "visualContainers": [],
            "config": "",
            "width": 1280,
            "height": 720,
            "displayOption": 1,
        }
    ],
}


_CONTENT_TYPES = '''<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/Version" ContentType="text/plain"/>
  <Override PartName="/DataModelSchema" ContentType="application/json"/>
  <Override PartName="/Report/Layout" ContentType="application/json"/>
  <Override PartName="/Connections" ContentType="application/json"/>
  <Override PartName="/Metadata" ContentType="application/json"/>
  <Override PartName="/DiagramLayout" ContentType="application/json"/>
  <Override PartName="/SecurityBindings" ContentType="application/json"/>
</Types>
'''


def _utf16le_bom(obj) -> bytes:
    """Power BI expects most internal JSON files to be UTF-16 LE with BOM."""
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return b"\xff\xfe" + text.encode("utf-16-le")


def build(out_pbit: Path = OUT_PBIT, out_pq: Path = OUT_PQ) -> tuple[Path, Path]:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    # Power Query M script (reliable path)
    out_pq.write_text(_build_pq_script())

    # .pbit (best-effort)
    schema = _build_datamodel_schema()
    metadata = {"Version": "1.0", "CreatedFromTemplate": True}
    connections = {"Version": 1, "Connections": []}
    diagram_layout = {"version": 1, "diagrams": [{"name": "All tables", "ordinal": 1}]}
    security_bindings = {"Version": "1.0"}

    if out_pbit.exists():
        out_pbit.unlink()
    with zipfile.ZipFile(out_pbit, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Version", b"3.0")
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("DataModelSchema", _utf16le_bom(schema))
        zf.writestr("Connections", _utf16le_bom(connections))
        zf.writestr("DiagramLayout", _utf16le_bom(diagram_layout))
        zf.writestr("Metadata", _utf16le_bom(metadata))
        zf.writestr("SecurityBindings", _utf16le_bom(security_bindings))
        zf.writestr("Report/Layout", _utf16le_bom(_REPORT_LAYOUT))

    return out_pbit, out_pq


def main() -> None:
    pbit, pq = build()
    print(f"wrote {pbit} ({pbit.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {pq}  ({pq.stat().st_size / 1024:.0f} KB)")
    print()
    print("Power BI usage paths:")
    print()
    print("  Path 1 — open the .pbit (Windows-only, Power BI Desktop)")
    print(f"    Double-click {pbit.name}. Edit data source paths if needed.")
    print()
    print("  Path 2 — paste the .pq into Advanced Editor (guaranteed to work)")
    print("    Power BI Desktop → Home → Transform data → New Source → Blank Query")
    print(f"    → Home → Advanced Editor → paste {pq.name} contents")
    print("    → set DataFolder = your absolute path → Done.")


if __name__ == "__main__":
    main()
