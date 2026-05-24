"""Energy / efficiency KPIs — the IREN-specific angle.

A renewable-powered data center treats wasted energy as a first-class
operational metric, not a sustainability talking point. These are the
panels that will get the dashboard noticed.
"""
from __future__ import annotations

import pandas as pd

# H100-class assumptions; tune as you like.
GPU_TDP_WATTS = 700.0
NODE_OVERHEAD_WATTS = 600.0  # CPU + memory + fans + NIC
GPU_PEAK_TFLOPS_FP16 = 1000.0  # approx


def node_power_watts(gpu_utils: list[float]) -> float:
    """Linear power model: idle ~30% TDP, full load ~100% TDP. Plus node overhead."""
    return NODE_OVERHEAD_WATTS + sum(GPU_TDP_WATTS * (0.3 + 0.7 * u) for u in gpu_utils)


def perf_per_watt(gpu_utils: pd.Series, power_kw: pd.Series) -> float:
    """Aggregate TFLOPS / total kW.   Series indexed identically by tick."""
    if power_kw.sum() == 0:
        return 0.0
    tflops = (gpu_utils * GPU_PEAK_TFLOPS_FP16).sum()
    return float(tflops / (power_kw.sum() * 1000))  # tflops per W


def idle_energy_waste_kwh(util_df: pd.DataFrame, idle_threshold: float = 0.05) -> float:
    """Sum kWh spent at < idle_threshold utilization.

    util_df cols: tick, gpu_id, util. Assumes one sample per hour.
    """
    idle = util_df[util_df["util"] < idle_threshold]
    if idle.empty:
        return 0.0
    # at idle, each GPU still draws ~0.3 × TDP
    idle_watts = len(idle) * 0.3 * GPU_TDP_WATTS
    return idle_watts / 1000  # 1-hour samples → kWh


def pue(total_facility_kw: float, it_kw: float) -> float:
    """Power Usage Effectiveness. 1.0 is theoretical perfect; ~1.1-1.2 is best-in-class."""
    if it_kw <= 0:
        return 0.0
    return total_facility_kw / it_kw


def renewable_match(renewable_kwh: float, total_kwh: float) -> float:
    """Fraction of consumed energy matched by renewable supply."""
    if total_kwh <= 0:
        return 0.0
    return min(1.0, renewable_kwh / total_kwh)


def idle_waste_dollars(idle_kwh: float, price_per_kwh: float = 0.06) -> float:
    """Annualized $ cost of idle GPU draw — the headline interview number."""
    return idle_kwh * price_per_kwh


def power_timeseries(util_df: pd.DataFrame, gpus_per_node: int = 8) -> pd.DataFrame:
    """Aggregate utilization parquet → per-tick fleet power and GPU-hour series.

    Input cols: tick, gpu_id, util (one row per GPU per sample tick).
    Output cols: tick, fleet_kw, active_gpu_count, gpu_hours_in_tick.

    Each util sample represents ONE HOUR of GPU time (emit.py samples once per
    `ticks_per_hour`). So gpu_hours_in_tick == util.sum() per tick.
    """
    if util_df.empty:
        return pd.DataFrame(columns=["tick", "fleet_kw", "active_gpu_count", "gpu_hours_in_tick"])

    per_tick = util_df.groupby("tick").agg(
        mean_util=("util", "mean"),
        active_gpu_count=("util", lambda s: int((s > 0.05).sum())),
        gpu_hours_in_tick=("util", "sum"),
        gpu_count=("util", "count"),
    )
    # node-overhead is per node; util_df has one row per GPU
    n_nodes = (per_tick["gpu_count"] / gpus_per_node).round().astype(int)
    gpu_w_total = (per_tick["mean_util"] * GPU_TDP_WATTS * 0.7 + GPU_TDP_WATTS * 0.3) * per_tick[
        "gpu_count"
    ]
    node_w_total = n_nodes * NODE_OVERHEAD_WATTS
    per_tick["fleet_kw"] = (gpu_w_total + node_w_total) / 1000.0
    return per_tick.reset_index()[["tick", "fleet_kw", "active_gpu_count", "gpu_hours_in_tick"]]


def fleet_energy_summary(
    util_df: pd.DataFrame,
    gpus_per_node: int = 8,
    idle_threshold: float = 0.05,
    price_per_kwh: float = 0.06,
) -> dict[str, float]:
    """One-shot summary for the dashboard's energy panel."""
    ts = power_timeseries(util_df, gpus_per_node=gpus_per_node)
    if ts.empty:
        return {}

    total_kwh = float(ts["fleet_kw"].sum())  # 1-hour samples → kW × 1h = kWh
    idle_kwh = idle_energy_waste_kwh(util_df, idle_threshold=idle_threshold)
    active_gpu_hours = float(util_df.loc[util_df["util"] >= idle_threshold, "util"].sum())
    total_gpu_hours = float(util_df["util"].sum())
    aggregate_tflops = total_gpu_hours * GPU_PEAK_TFLOPS_FP16
    avg_kw = float(ts["fleet_kw"].mean())
    perf_w = aggregate_tflops / (total_kwh * 1000.0) if total_kwh > 0 else 0.0  # TFLOPS/W

    return {
        "total_kwh": total_kwh,
        "idle_kwh": idle_kwh,
        "idle_waste_dollars": idle_waste_dollars(idle_kwh, price_per_kwh),
        "active_gpu_hours": active_gpu_hours,
        "total_gpu_hours": total_gpu_hours,
        "avg_fleet_kw": avg_kw,
        "perf_per_watt_tflops": perf_w,
    }
