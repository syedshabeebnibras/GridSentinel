"""Generate a .twbx using Tableau's official Hyper extract format.

Hyper is Tableau's native columnar database (replaces TDE files). Workbooks
that reference .hyper extracts load instantly and bypass the entire CSV
text-scanning code path that has been causing 501CF476 errors with our
hand-written textscan datasources.

Pipeline:
  1. CSV → pandas DataFrame
  2. DataFrame → .hyper file (one per table, written via tableauhyperapi)
  3. Hyper files + minimal workbook XML → .twbx zip

Workbook XML still has zero worksheets / dashboards — those still need to
be authored visually in Tableau Public Desktop after opening.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from textwrap import dedent

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLEAU_SRC = REPO_ROOT / "data" / "tableau"
OUT_FILE = TABLEAU_SRC / "GridSentinel.twbx"
TMP_HYPER_DIR = REPO_ROOT / "data" / "tableau" / "_hyper"


_TABLES = [
    "fact_events",
    "fact_incidents",
    "fact_hourly_kpi",
    "fact_clusters",
    "dim_fleet",
    "dim_time",
]


def _csv_to_hyper(csv_path: Path, hyper_path: Path) -> None:
    from tableauhyperapi import (
        Connection,
        CreateMode,
        HyperProcess,
        SqlType,
        TableDefinition,
        Telemetry,
        Inserter,
        Nullability,
        TableName,
    )

    df = pd.read_csv(csv_path)

    def _column_kind(series: pd.Series) -> str:
        dt = series.dtype
        if pd.api.types.is_bool_dtype(dt):
            return "bool"
        if pd.api.types.is_integer_dtype(dt):
            return "int"
        if pd.api.types.is_float_dtype(dt):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(dt):
            return "datetime"
        return "text"

    column_kinds = {col: _column_kind(df[col]) for col in df.columns}

    def _to_sql_type(kind: str) -> SqlType:
        return {
            "bool": SqlType.bool(),
            "int": SqlType.big_int(),
            "float": SqlType.double(),
            "datetime": SqlType.timestamp(),
            "text": SqlType.text(),
        }[kind]

    columns = [
        TableDefinition.Column(
            name=col,
            type=_to_sql_type(column_kinds[col]),
            nullability=Nullability.NULLABLE,
        )
        for col in df.columns
    ]
    table_def = TableDefinition(
        table_name=TableName("Extract", csv_path.stem),
        columns=columns,
    )

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        if hyper_path.exists():
            hyper_path.unlink()
        with Connection(
            endpoint=hyper.endpoint,
            database=hyper_path,
            create_mode=CreateMode.CREATE_AND_REPLACE,
        ) as conn:
            conn.catalog.create_schema("Extract")
            conn.catalog.create_table(table_def)
            with Inserter(conn, table_def) as inserter:
                rows: list[list] = []
                for _, row in df.iterrows():
                    out: list = []
                    for col in df.columns:
                        v = row[col]
                        if pd.isna(v):
                            out.append(None)
                            continue
                        kind = column_kinds[col]
                        if kind == "int":
                            out.append(int(v))
                        elif kind == "float":
                            out.append(float(v))
                        elif kind == "bool":
                            out.append(bool(v))
                        elif kind == "datetime":
                            out.append(pd.Timestamp(v).to_pydatetime())
                        else:
                            out.append(str(v))
                    rows.append(out)
                inserter.add_rows(rows)
                inserter.execute()


def _datasource_xml(table: str) -> str:
    """Workbook datasource pointing at a bundled .hyper extract."""
    safe_caption = table.replace("_", " ").title()
    return dedent(f'''\
    <datasource caption="{safe_caption}" inline="true" name="federated.{table}" version="18.1">
      <connection class="federated">
        <named-connections>
          <named-connection caption="{table}.hyper" name="hyper.{table}">
            <connection class="hyper" dbname="Data/Datasources/{table}.hyper" username="tableau_internal_user" />
          </named-connection>
        </named-connections>
        <relation connection="hyper.{table}" name="{table}" table="[Extract].[{table}]" type="table" />
      </connection>
    </datasource>''')


def _build_twb_xml() -> str:
    sources = "\n".join(_datasource_xml(t) for t in _TABLES)
    return dedent(f'''\
<?xml version='1.0' encoding='utf-8' ?>
<!-- GridSentinel — Tableau workbook using Hyper extracts -->
<workbook source-build="2024.1.0" source-platform="mac" version="18.1">
  <preferences />
  <datasources>
{sources}
  </datasources>
</workbook>
''')


def build(out_file: Path = OUT_FILE) -> Path:
    if not TABLEAU_SRC.exists() or not list(TABLEAU_SRC.glob("*.csv")):
        raise FileNotFoundError(
            "No CSVs in data/tableau/. Run `python -m gridsentinel.tableau.export` first."
        )

    if TMP_HYPER_DIR.exists():
        shutil.rmtree(TMP_HYPER_DIR)
    TMP_HYPER_DIR.mkdir(parents=True, exist_ok=True)

    for csv in sorted(TABLEAU_SRC.glob("*.csv")):
        hyper_path = TMP_HYPER_DIR / (csv.stem + ".hyper")
        print(f"  converting {csv.name} → {hyper_path.name}…", flush=True)
        _csv_to_hyper(csv, hyper_path)

    twb_xml = _build_twb_xml()
    if out_file.exists():
        out_file.unlink()

    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("GridSentinel.twb", twb_xml)
        for h in sorted(TMP_HYPER_DIR.glob("*.hyper")):
            zf.write(h, f"Data/Datasources/{h.name}")

    shutil.rmtree(TMP_HYPER_DIR, ignore_errors=True)
    return out_file


def main() -> None:
    p = build()
    size_mb = p.stat().st_size / (1024 * 1024)
    print()
    print(f"wrote {p} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
