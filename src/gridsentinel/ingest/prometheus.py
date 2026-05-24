"""Prometheus / Mimir / VictoriaMetrics ingestion.

Two paths supported:
  1. Snapshot file produced by `curl /api/v1/query_range?...` → JSON
  2. Live HTTP query — only used when a PROMETHEUS_URL env var is set; we
     ship a thin wrapper that uses urllib (no extra deps).

The query_range API returns:
    {
      "status": "success",
      "data": {
        "resultType": "matrix",
        "result": [
          {
            "metric": {"__name__": "DCGM_FI_DEV_GPU_TEMP", "gpu": "0", "hostname": "node-0001", ...},
            "values": [[1700000000, "62.3"], [1700000060, "62.4"], ...]
          },
          ...
        ]
      }
    }

We melt that into the long format and pivot to our standard wide schema.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from gridsentinel.ingest.dcgm import TARGET_COLUMNS

# Map Prometheus DCGM metric names to our schema columns.
_METRIC_TO_COLUMN = {
    "DCGM_FI_DEV_GPU_TEMP": "gpu_temp_c",
    "DCGM_FI_DEV_POWER_USAGE": "gpu_power_w",
    "DCGM_FI_DEV_GPU_UTIL": "gpu_sm_util",
    "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "ecc_corrected_total",
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": "nvlink_crc_total",
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": "pcie_aer_total",
}


def parse_query_range(payload: dict, ticks_per_hour: int = 12) -> pd.DataFrame:
    """Convert a Prometheus query_range response to our long-then-wide schema."""
    if payload.get("status") != "success":
        raise ValueError(f"prometheus query failed: {payload.get('error', 'unknown')}")
    result = payload.get("data", {}).get("result", [])
    if not result:
        return pd.DataFrame(columns=list(TARGET_COLUMNS))

    long_rows = []
    for series in result:
        labels = series.get("metric", {})
        name = labels.get("__name__")
        col = _METRIC_TO_COLUMN.get(name)
        if col is None:
            continue
        node_id = labels.get("hostname") or labels.get("Hostname") or labels.get("instance", "")
        gpu_idx = labels.get("gpu") or labels.get("GPU-Id") or "0"
        gpu_id = f"{node_id}/gpu{gpu_idx}"
        for ts_s, val_s in series.get("values", []):
            try:
                long_rows.append({
                    "ts": float(ts_s),
                    "gpu_id": gpu_id,
                    "node_id": node_id,
                    "gpu_idx": int(gpu_idx),
                    "metric": col,
                    "value": float(val_s),
                })
            except (ValueError, TypeError):
                continue
    if not long_rows:
        return pd.DataFrame(columns=list(TARGET_COLUMNS))

    long = pd.DataFrame(long_rows)
    wide = (
        long.pivot_table(
            index=["ts", "gpu_id", "node_id", "gpu_idx"],
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )

    # GPU util in Prometheus is 0..100; normalise to 0..1
    if "gpu_sm_util" in wide.columns and wide["gpu_sm_util"].max() > 1.5:
        wide["gpu_sm_util"] = wide["gpu_sm_util"] / 100.0

    # Derive `tick` from `ts` at the sample cadence implied by ticks_per_hour.
    ts0 = wide["ts"].min()
    seconds_per_tick = 3600 / ticks_per_hour
    wide["tick"] = ((wide["ts"] - ts0) / seconds_per_tick).round().astype(int)

    wide["rack_id"] = None
    wide["zone_id"] = None
    for col in TARGET_COLUMNS:
        if col not in wide.columns:
            wide[col] = 0
    return wide[list(TARGET_COLUMNS)].reset_index(drop=True)


def from_file(path: str | Path, ticks_per_hour: int = 12) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    return parse_query_range(payload, ticks_per_hour=ticks_per_hour)


def from_http(
    metric_names: list[str] | None = None,
    start: float | None = None,
    end: float | None = None,
    step_seconds: int = 60,
    base_url: str | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Query a live Prometheus / Mimir / VictoriaMetrics endpoint.

    Set PROMETHEUS_URL env var or pass `base_url=`.   No extra deps — uses
    stdlib urllib. This is the path for connecting GridSentinel to a real
    ops Prometheus instance.
    """
    base_url = base_url or os.environ.get("PROMETHEUS_URL")
    if not base_url:
        raise ValueError("set PROMETHEUS_URL or pass base_url=")
    if metric_names is None:
        metric_names = list(_METRIC_TO_COLUMN.keys())

    import time
    if end is None:
        end = time.time()
    if start is None:
        start = end - 14 * 24 * 3600  # default: last 14 days

    out_payload = {"status": "success", "data": {"resultType": "matrix", "result": []}}
    for m in metric_names:
        params = urllib.parse.urlencode({
            "query": m,
            "start": start,
            "end": end,
            "step": step_seconds,
        })
        url = f"{base_url.rstrip('/')}/api/v1/query_range?{params}"
        with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode())
        if payload.get("status") == "success":
            out_payload["data"]["result"].extend(payload["data"]["result"])
    return parse_query_range(out_payload)
