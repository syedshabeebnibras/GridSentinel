import pandas as pd

from gridsentinel.kpi.efficiency import (
    fleet_energy_summary,
    perf_per_watt,
    power_timeseries,
)


def _util_df():
    # 16 GPUs (= 2 nodes × 8) sampled at 2 ticks
    rows = []
    for tick in [0, 1]:
        for n in range(2):
            for g in range(8):
                rows.append(
                    {
                        "tick": tick,
                        "gpu_id": f"node-{n:04d}/gpu{g}",
                        "util": 0.5 if (tick + g) % 2 == 0 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_power_timeseries_shapes():
    ts = power_timeseries(_util_df(), gpus_per_node=8)
    assert {"tick", "fleet_kw", "active_gpu_count", "gpu_hours_in_tick"} <= set(ts.columns)
    assert len(ts) == 2
    assert (ts["fleet_kw"] > 0).all()


def test_fleet_energy_summary_keys():
    summary = fleet_energy_summary(_util_df(), gpus_per_node=8)
    expected = {
        "total_kwh",
        "idle_kwh",
        "idle_waste_dollars",
        "active_gpu_hours",
        "total_gpu_hours",
        "avg_fleet_kw",
        "perf_per_watt_tflops",
    }
    assert expected <= set(summary)
    assert summary["total_kwh"] > 0


def test_perf_per_watt_zero_on_zero_power():
    gpu_utils = pd.Series([0.5, 0.5])
    power = pd.Series([0.0, 0.0])
    assert perf_per_watt(gpu_utils, power) == 0.0
