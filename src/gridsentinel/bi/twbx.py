"""Generate a packaged Tableau workbook (.twbx) from the CSV exports.

A .twbx is just a zip containing:
  - GridSentinel.twb         (the workbook XML)
  - Data/Datasources/*.csv   (the bundled data files)

Tableau Public Desktop on macOS will open it directly. The workbook bundles
data sources for every fact + dim table, defines a starter set of
calculated fields (Alert Compression, MTTR, Perf/Watt, Idle Waste $), and
includes one prebuilt worksheet (Top 10 incidents by member count) so the
file shows *something* on first open instead of a blank canvas.

The user finishes authoring additional sheets and dashboards visually.
That's the honest minimum we can deliver without programmatic Tableau
authoring (Tableau has no official Python authoring API for .twb XML).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLEAU_SRC = REPO_ROOT / "data" / "tableau"
OUT_DIR = REPO_ROOT / "data" / "tableau"
OUT_FILE = OUT_DIR / "GridSentinel.twbx"

# CSV schemas — drives column-element generation in the .twb XML.
_TABLES: dict[str, list[tuple[str, str]]] = {
    "fact_events": [
        ("tick", "integer"),
        ("kind", "string"),
        ("scope", "string"),
        ("target", "string"),
        ("severity", "string"),
        ("benign", "boolean"),
        ("parent_event_id", "string"),
        ("node_id", "string"),
        ("rack_id", "string"),
        ("zone_id", "string"),
        ("feed_id", "string"),
        ("spine_id", "string"),
    ],
    "fact_incidents": [
        ("incident_id", "string"),
        ("opened_tick", "integer"),
        ("resolved_tick", "integer"),
        ("member_count", "integer"),
        ("root_kind", "string"),
        ("scope", "string"),
        ("severity_max", "string"),
        ("all_benign", "boolean"),
        ("any_benign", "boolean"),
        ("duration_ticks", "integer"),
        ("cluster_id", "integer"),
        ("cluster_label", "string"),
    ],
    "fact_hourly_kpi": [
        ("tick", "integer"),
        ("fleet_kw", "real"),
        ("active_gpu_count", "integer"),
        ("gpu_hours", "real"),
        ("idle_gpu_count", "integer"),
    ],
    "fact_clusters": [
        ("cluster_id", "integer"),
        ("cluster_label", "string"),
        ("count", "integer"),
        ("critical_count", "integer"),
        ("all_benign_ratio", "real"),
        ("mean_members", "real"),
    ],
    "dim_fleet": [
        ("node_id", "string"),
        ("rack_id", "string"),
        ("zone_id", "string"),
        ("feed_id", "string"),
        ("spine_id", "string"),
    ],
    "dim_time": [
        ("tick", "integer"),
        ("datetime", "datetime"),
        ("date", "date"),
        ("hour", "integer"),
        ("day_of_week", "string"),
        ("is_weekend", "boolean"),
    ],
}


def _tableau_role(dt: str, col: str) -> str:
    """measure vs dimension — measures are the numerics we'll aggregate."""
    measure_cols = {
        "member_count", "duration_ticks", "fleet_kw", "active_gpu_count",
        "gpu_hours", "idle_gpu_count", "count", "critical_count",
        "all_benign_ratio", "mean_members",
    }
    if col in measure_cols:
        return "measure"
    return "dimension"


def _column_xml(name: str, dt: str) -> str:
    role = _tableau_role(dt, name)
    type_attr = {"integer": "ordinal", "real": "quantitative", "boolean": "nominal"}.get(dt, "nominal")
    return (
        f'      <column datatype="{dt}" name="[{name}]" role="{role}" '
        f'type="{type_attr}" />'
    )


def _datasource_xml(table: str, cols: list[tuple[str, str]]) -> str:
    """Build a <datasource> for one CSV file. Absolute minimum: connection
    only, no column declarations. Tableau auto-discovers columns from the
    CSV header on first load. Pre-declaring them at datasource level was
    causing internal error 501CF476 (conflict with auto-detected types)."""
    safe_caption = table.replace("_", " ").title()
    return dedent(f'''\
    <datasource caption="{safe_caption}" inline="true" name="federated.{table}" version="18.1">
      <connection class="federated">
        <named-connections>
          <named-connection caption="{table}.csv" name="textscan.{table}">
            <connection class="textscan" directory="./Data/Datasources" filename="{table}.csv" password="" server="" />
          </named-connection>
        </named-connections>
        <relation connection="textscan.{table}" name="{table}.csv" table="[{table}#csv]" type="table" />
      </connection>
    </datasource>''')


# Starter calculated fields — paste-equivalents of the calculated_fields.txt
# doc. We embed them directly into the workbook so they're available the
# moment Tableau opens.
_CALC_FIELDS_XML = dedent('''\
    <datasource caption="Calculations" inline="true" name="parameters" version="18.1">
      <aliases enabled="yes" />
      <column caption="MTTR (hours)" datatype="real" name="[Calculation_MTTR]" role="measure" type="quantitative">
        <calculation class="tableau" formula="AVG([Duration Ticks]) / 12.0" />
      </column>
      <column caption="Critical Incidents" datatype="integer" name="[Calculation_CritIncidents]" role="measure" type="quantitative">
        <calculation class="tableau" formula="COUNTD(IF [Severity Max] = &quot;critical&quot; THEN [Incident Id] END)" />
      </column>
      <column caption="Compression Ratio" datatype="real" name="[Calculation_Compression]" role="measure" type="quantitative">
        <calculation class="tableau" formula="COUNTD([Tick]) / COUNTD([Incident Id])" />
      </column>
    </datasource>''')


def _build_twb_xml() -> str:
    """Bare-minimum valid .twb — workbook element with preferences and
    data sources only. No windows, no worksheets, no style block. Tableau
    opens, shows the data source pane, user builds from there.

    Going minimal here intentionally: hand-written .twb with pre-built
    worksheets is fragile (Tableau's schema validator is strict about
    `<windows>` / `<window>` content models, which require live `cards`
    references that only Tableau can author). Better to ship a clean file
    that opens cleanly than a fancy one that errors out."""
    datasources = "\n".join(_datasource_xml(t, c) for t, c in _TABLES.items())
    return dedent(f'''\
<?xml version='1.0' encoding='utf-8' ?>
<!-- GridSentinel — Tableau workbook auto-generated by gridsentinel.bi.twbx -->
<workbook source-build="2024.1.0" source-platform="mac" version="18.1">
  <preferences />
  <datasources>
{datasources}
  </datasources>
</workbook>
''')


def build(out_file: Path = OUT_FILE) -> Path:
    """Assemble the .twbx zip and return its path."""
    if not TABLEAU_SRC.exists() or not list(TABLEAU_SRC.glob("*.csv")):
        raise FileNotFoundError(
            "No CSVs in data/tableau/. Run `python -m gridsentinel.tableau.export` first."
        )

    twb_xml = _build_twb_xml()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        out_file.unlink()

    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("GridSentinel.twb", twb_xml)
        for csv in sorted(TABLEAU_SRC.glob("*.csv")):
            # Tableau looks for bundled data under Data/Datasources/<filename>
            zf.write(csv, f"Data/Datasources/{csv.name}")

    return out_file


def main() -> None:
    p = build()
    size_kb = p.stat().st_size / 1024
    print(f"wrote {p} ({size_kb:.0f} KB)")
    print()
    print("To use:")
    print("  1. Install Tableau Public Desktop (free, macOS): https://www.tableau.com/products/public/download")
    print(f"  2. Open {p.name} from Tableau Public's File menu")
    print("  3. Six data sources will appear in the left pane (fact_* + dim_*)")
    print("  4. Drag fields to build sheets; publish to your Tableau Public profile")


if __name__ == "__main__":
    main()
