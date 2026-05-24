"""OpenTelemetry OTLP/JSON metrics ingestion.

OTLP is the OpenTelemetry wire format. Most modern fleets emit GPU metrics
via OpenTelemetry collectors (`otelcol` + `nvidia-otel-receiver`), which
forward to a backend in OTLP/JSON. Schema:

    {
      "resourceMetrics": [{
        "resource": {"attributes": [{"key": "host.name", "value": {"stringValue": "node-0001"}}, ...]},
        "scopeMetrics": [{
          "metrics": [{
            "name": "gpu.temperature",
            "gauge": {
              "dataPoints": [{
                "asDouble": 62.3,
                "timeUnixNano": "1700000000000000000",
                "attributes": [{"key": "gpu.id", "value": {"stringValue": "0"}}]
              }, ...]
            }
          }, ...]
        }]
      }]
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridsentinel.ingest.dcgm import TARGET_COLUMNS

_OTEL_METRIC_MAP = {
    "gpu.temperature": "gpu_temp_c",
    "gpu.power.draw": "gpu_power_w",
    "gpu.utilization": "gpu_sm_util",
    "gpu.memory.ecc.errors.corrected": "ecc_corrected_total",
    "gpu.nvlink.crc.errors": "nvlink_crc_total",
    "gpu.pcie.replay.count": "pcie_aer_total",
}


def _attr_value(attr: dict):
    """Extract the actual value from an OTLP attribute (multiple value-type keys possible)."""
    v = attr.get("value", {})
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return v[k]
    return None


def _attrs_to_dict(attrs: list[dict]) -> dict:
    return {a["key"]: _attr_value(a) for a in (attrs or [])}


def parse_otlp(payload: dict, ticks_per_hour: int = 12) -> pd.DataFrame:
    rows = []
    for rm in payload.get("resourceMetrics", []):
        resource_attrs = _attrs_to_dict(rm.get("resource", {}).get("attributes", []))
        node_id = (
            resource_attrs.get("host.name")
            or resource_attrs.get("k8s.node.name")
            or resource_attrs.get("service.instance.id")
            or "node-unknown"
        )
        for sm in rm.get("scopeMetrics", []):
            for metric in sm.get("metrics", []):
                name = metric.get("name")
                col = _OTEL_METRIC_MAP.get(name)
                if col is None:
                    continue
                data_points = (
                    metric.get("gauge", {}).get("dataPoints")
                    or metric.get("sum", {}).get("dataPoints")
                    or []
                )
                for dp in data_points:
                    dp_attrs = _attrs_to_dict(dp.get("attributes", []))
                    gpu_idx = dp_attrs.get("gpu.id") or dp_attrs.get("gpu") or "0"
                    val = dp.get("asDouble")
                    if val is None and "asInt" in dp:
                        val = float(dp["asInt"])
                    if val is None:
                        continue
                    rows.append({
                        "ts_ns": int(dp.get("timeUnixNano", 0)),
                        "node_id": str(node_id),
                        "gpu_idx": int(gpu_idx),
                        "gpu_id": f"{node_id}/gpu{gpu_idx}",
                        "metric": col,
                        "value": float(val),
                    })
    if not rows:
        return pd.DataFrame(columns=list(TARGET_COLUMNS))

    long = pd.DataFrame(rows)
    wide = (
        long.pivot_table(
            index=["ts_ns", "gpu_id", "node_id", "gpu_idx"],
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    if "gpu_sm_util" in wide.columns and wide["gpu_sm_util"].max() > 1.5:
        wide["gpu_sm_util"] = wide["gpu_sm_util"] / 100.0

    ts0 = wide["ts_ns"].min()
    ns_per_tick = (3600 / ticks_per_hour) * 1e9
    wide["tick"] = ((wide["ts_ns"] - ts0) / ns_per_tick).round().astype(int)

    wide["rack_id"] = None
    wide["zone_id"] = None
    for col in TARGET_COLUMNS:
        if col not in wide.columns:
            wide[col] = 0
    return wide[list(TARGET_COLUMNS)].reset_index(drop=True)


def from_file(path: str | Path, ticks_per_hour: int = 12) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    return parse_otlp(payload, ticks_per_hour=ticks_per_hour)
