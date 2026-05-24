"""DCGM CSV import — adapt NVIDIA Data Center GPU Manager exports to our schema.

`dcgmi dmon -e <field_ids> -c <count>` produces a fixed-width text stream
with one row per (gpu_id, timestamp). Modern fleet ops typically pipe this
through Prometheus, but the raw CSV path is still common in air-gapped
environments and is the simplest real-world ingest to demonstrate.

Field-id mapping we care about (from `dcgmi dmon -l`):

    150 GPUTL  → gpu_sm_util            (0..100, we / 100 → 0..1)
    155 POWER  → gpu_power_w
    140 GTEMP  → gpu_temp_c
    310 ECCAR  → ecc_corrected_total
    449 NVLBE  → nvlink_crc_total       (sum across links)
    450 PCIRX  → pcie_aer_total proxy   (RX retry counter; AER not directly exposed)

This module accepts both the raw `dcgmi dmon` whitespace-delimited format
*and* the more common Prometheus-friendly CSV variant. The output schema
matches data/synthetic/metrics.parquet exactly so downstream code is
indifferent to whether the data is synthetic or real.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

# Schema we promise to produce — same as simulator/metrics.py output.
TARGET_COLUMNS = (
    "tick", "gpu_id", "node_id", "rack_id", "zone_id",
    "gpu_temp_c", "gpu_power_w", "gpu_sm_util",
    "ecc_corrected_total", "nvlink_crc_total", "pcie_aer_total",
)


def _parse_dcgmi_dmon(text: str) -> pd.DataFrame:
    """Parse the whitespace-delimited `dcgmi dmon` output.

    Format:
        #Entity   GPUTL POWER GTEMP ECCAR NVLBE PCIRX
        GPU 0       42   175    62     0    13    0
        GPU 1       38   170    61     0    11    0
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=list(TARGET_COLUMNS))
    # Skip comment / header lines starting with '#'
    rows = []
    for ln in lines:
        if ln.startswith("#") or "Entity" in ln or "GPU-Id" in ln:
            continue
        parts = ln.split()
        if len(parts) < 7 or parts[0] not in ("GPU", "gpu"):
            continue
        gpu_idx = int(parts[1])
        try:
            rows.append(
                {
                    "gpu_idx": gpu_idx,
                    "gpu_sm_util": float(parts[2]) / 100.0,
                    "gpu_power_w": float(parts[3]),
                    "gpu_temp_c": float(parts[4]),
                    "ecc_corrected_total": int(parts[5]),
                    "nvlink_crc_total": int(parts[6]),
                    "pcie_aer_total": int(parts[7]) if len(parts) > 7 else 0,
                }
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def _parse_dcgm_csv(text: str) -> pd.DataFrame:
    """Parse the Prometheus-style CSV that dcgm-exporter writes.

    Expected columns include: timestamp, gpu_id, hostname (= our node_id),
    plus metric columns prefixed `DCGM_FI_*`.
    """
    df = pd.read_csv(StringIO(text))
    rename = {
        "DCGM_FI_DEV_GPU_TEMP": "gpu_temp_c",
        "DCGM_FI_DEV_POWER_USAGE": "gpu_power_w",
        "DCGM_FI_DEV_GPU_UTIL": "gpu_sm_util",
        "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "ecc_corrected_total",
        "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": "nvlink_crc_total",
        "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": "pcie_aer_total",
        "Hostname": "node_id",
        "hostname": "node_id",
        "host": "node_id",
        "GPU-Id": "gpu_idx",
        "gpu": "gpu_idx",
        "timestamp": "ts",
        "Timestamp": "ts",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "gpu_sm_util" in df.columns and df["gpu_sm_util"].max() > 1.5:
        df["gpu_sm_util"] = df["gpu_sm_util"] / 100.0
    return df


def from_dcgm(
    source: str | Path,
    fleet: pd.DataFrame | None = None,
    ticks_per_hour: int = 12,
) -> pd.DataFrame:
    """Read a DCGM export, auto-detect format, return our standard schema.

    Args:
        source: file path or raw text.
        fleet: optional fleet dimension table to enrich with rack/zone/feed.
               If absent we infer node_id from gpu_id where possible and
               leave topology columns empty (downstream merge can fill them).
    """
    text: str
    if isinstance(source, Path):
        text = source.read_text()
    elif isinstance(source, str):
        # treat as file path only if it's short and looks pathy; otherwise raw text
        looks_like_path = (
            len(source) < 4096
            and "\n" not in source
            and ("/" in source or source.endswith(".csv") or source.endswith(".txt"))
        )
        if looks_like_path:
            try:
                p = Path(source)
                if p.exists():
                    text = p.read_text()
                else:
                    text = source
            except OSError:
                text = source
        else:
            text = source
    else:
        text = str(source)

    # Auto-detect: CSV if first non-empty line has commas; otherwise dmon.
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    df = _parse_dcgm_csv(text) if "," in first_line else _parse_dcgmi_dmon(text)
    if df.empty:
        return pd.DataFrame(columns=list(TARGET_COLUMNS))

    # Derive tick from timestamp if available (round to sample interval).
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"])
        # tick = minutes since first sample → /5 to get 5-min cadence at
        # ticks_per_hour=12 (each tick = 5 min).
        minutes = (ts - ts.min()).dt.total_seconds() / 60
        df["tick"] = (minutes / (60 / ticks_per_hour)).round().astype(int)
    else:
        df["tick"] = 0

    # Compose gpu_id and node_id where missing.
    if "gpu_id" not in df.columns:
        if "node_id" in df.columns and "gpu_idx" in df.columns:
            df["gpu_id"] = df["node_id"].astype(str) + "/gpu" + df["gpu_idx"].astype(str)
        elif "gpu_idx" in df.columns:
            df["gpu_id"] = "node-unknown/gpu" + df["gpu_idx"].astype(str)
    if "node_id" not in df.columns:
        df["node_id"] = df["gpu_id"].str.split("/").str[0]

    # Topology — join from fleet table if provided
    if fleet is not None and {"rack_id", "zone_id"}.issubset(fleet.columns):
        df = df.merge(
            fleet[["node_id", "rack_id", "zone_id"]],
            on="node_id", how="left",
        )
    else:
        df["rack_id"] = None
        df["zone_id"] = None

    # Ensure required columns present
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if "total" in col or "util" in col or col.endswith("_w") else None

    return df[list(TARGET_COLUMNS)].reset_index(drop=True)
