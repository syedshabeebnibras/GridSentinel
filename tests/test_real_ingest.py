"""Tests for the real-data ingest adapters (DCGM / Prometheus / OTel)."""
from __future__ import annotations

import json

import pandas as pd

from gridsentinel.ingest.dcgm import TARGET_COLUMNS, from_dcgm
from gridsentinel.ingest.otel import parse_otlp
from gridsentinel.ingest.prometheus import parse_query_range


def test_dcgm_dmon_format_parses():
    raw = """#Entity   GPUTL POWER GTEMP ECCAR NVLBE PCIRX
GPU 0       42   175    62     3    13     1
GPU 1       38   170    61     1    11     0
"""
    df = from_dcgm(raw)
    assert set(TARGET_COLUMNS) <= set(df.columns)
    assert len(df) == 2
    assert df.iloc[0]["gpu_temp_c"] == 62.0
    assert df.iloc[0]["gpu_sm_util"] == 0.42
    assert df.iloc[1]["gpu_power_w"] == 170.0


def test_dcgm_csv_format_parses():
    raw = (
        "timestamp,Hostname,GPU-Id,DCGM_FI_DEV_GPU_TEMP,DCGM_FI_DEV_POWER_USAGE,"
        "DCGM_FI_DEV_GPU_UTIL,DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,"
        "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,DCGM_FI_DEV_PCIE_REPLAY_COUNTER\n"
        "2026-05-01T00:00:00,node-0001,0,65.2,205,80,0,5,1\n"
        "2026-05-01T00:05:00,node-0001,0,67.1,215,85,0,6,1\n"
        "2026-05-01T00:00:00,node-0001,1,64.0,200,82,0,4,1\n"
    )
    df = from_dcgm(raw)
    assert set(TARGET_COLUMNS) <= set(df.columns)
    assert len(df) == 3
    assert df["gpu_sm_util"].max() <= 1.0  # normalised from 0..100
    # tick should be derivable from timestamp; first row tick should be 0
    assert (df["tick"] >= 0).all()
    # gpu_id derived from hostname + gpu-id
    assert "node-0001" in df["node_id"].iloc[0]


def test_prometheus_query_range_parses():
    payload = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {
                        "__name__": "DCGM_FI_DEV_GPU_TEMP",
                        "hostname": "node-0001",
                        "gpu": "0",
                    },
                    "values": [[1700000000, "62.3"], [1700000060, "62.4"]],
                },
                {
                    "metric": {
                        "__name__": "DCGM_FI_DEV_POWER_USAGE",
                        "hostname": "node-0001",
                        "gpu": "0",
                    },
                    "values": [[1700000000, "180.0"], [1700000060, "182.5"]],
                },
            ],
        },
    }
    df = parse_query_range(payload)
    assert set(TARGET_COLUMNS) <= set(df.columns)
    assert len(df) == 2
    assert "gpu0" in df["gpu_id"].iloc[0]
    assert df["gpu_temp_c"].iloc[0] == 62.3
    assert df["gpu_power_w"].iloc[1] == 182.5


def test_prometheus_handles_empty_result():
    payload = {"status": "success", "data": {"result": []}}
    df = parse_query_range(payload)
    assert df.empty


def test_otel_otlp_parses():
    payload = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {"key": "host.name", "value": {"stringValue": "node-0042"}}
                    ]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "gpu.temperature",
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "asDouble": 71.5,
                                            "timeUnixNano": "1700000000000000000",
                                            "attributes": [
                                                {"key": "gpu.id", "value": {"stringValue": "3"}}
                                            ],
                                        },
                                        {
                                            "asDouble": 72.1,
                                            "timeUnixNano": "1700000300000000000",
                                            "attributes": [
                                                {"key": "gpu.id", "value": {"stringValue": "3"}}
                                            ],
                                        },
                                    ]
                                },
                            },
                            {
                                "name": "gpu.power.draw",
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "asDouble": 410.0,
                                            "timeUnixNano": "1700000000000000000",
                                            "attributes": [
                                                {"key": "gpu.id", "value": {"stringValue": "3"}}
                                            ],
                                        }
                                    ]
                                },
                            },
                        ]
                    }
                ],
            }
        ]
    }
    df = parse_otlp(payload)
    assert set(TARGET_COLUMNS) <= set(df.columns)
    assert len(df) >= 1
    assert "node-0042" in df["node_id"].iloc[0]
    assert "gpu3" in df["gpu_id"].iloc[0]
    assert df["gpu_temp_c"].iloc[0] == 71.5
