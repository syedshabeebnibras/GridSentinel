"""Continuous DCGM-style telemetry — what a real fleet emits every minute.

Each GPU streams six signals, sampled at `metric_interval_ticks` cadence:

  - gpu_temp_c          : die temperature (°C)
  - gpu_power_w         : power draw (W)
  - gpu_sm_util         : streaming-multiprocessor activity (0..1)
  - ecc_corrected_total : monotonically increasing corrected-ECC counter
  - nvlink_crc_total    : monotonically increasing NVLink CRC error counter
  - pcie_aer_total      : monotonically increasing PCIe AER counter

The counters are the genuinely predictive signals — real PdM systems watch
their *rate of change*, not their instantaneous value. Counters jump well
before a critical event in our simulator, which gives time-series ML
something real to learn.

Output: data/synthetic/metrics.parquet (one row per (gpu, tick) sample).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from gridsentinel.simulator.topology import Fleet

AMBIENT_C = 22.0
TDP_W = 700.0
IDLE_FRACTION = 0.3  # idle GPU draws ~30% TDP


@dataclass
class GPUState:
    temp_c: float = 60.0
    ecc_corrected: int = 0
    nvlink_crc: int = 0
    pcie_aer: int = 0
    # latent fault flags — set by stateful failure models, decay over time
    ecc_pressure: float = 0.0
    nvlink_pressure: float = 0.0
    pcie_pressure: float = 0.0


@dataclass
class MetricsBus:
    """Container the failure-model functions write into.

    `failure_models.thermal_throttle`, `ecc_uncorrectable`, `nvlink_fault`,
    `pcie_error` each call `bump_*` to push the counters forward when their
    underlying stochastic event fires. The continuous-metrics emitter reads
    state from here every sampling tick.
    """
    states: dict[str, GPUState] = field(default_factory=dict)

    def get(self, gpu_id: str) -> GPUState:
        st = self.states.get(gpu_id)
        if st is None:
            st = GPUState()
            self.states[gpu_id] = st
        return st

    def bump_ecc(self, gpu_id: str, n: int = 1) -> None:
        st = self.get(gpu_id)
        st.ecc_corrected += n
        st.ecc_pressure = min(1.0, st.ecc_pressure + 0.3 * n)

    def bump_nvlink(self, gpu_id: str, n: int = 1) -> None:
        st = self.get(gpu_id)
        st.nvlink_crc += n
        st.nvlink_pressure = min(1.0, st.nvlink_pressure + 0.4 * n)

    def bump_pcie(self, gpu_id: str, n: int = 1) -> None:
        st = self.get(gpu_id)
        st.pcie_aer += n
        st.pcie_pressure = min(1.0, st.pcie_pressure + 0.35 * n)

    def decay(self, factor: float = 0.97) -> None:
        for st in self.states.values():
            st.ecc_pressure *= factor
            st.nvlink_pressure *= factor
            st.pcie_pressure *= factor


def sample_metrics(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    bus: MetricsBus,
    load_by_gpu: dict[str, float],
) -> list[dict]:
    """Emit one row per GPU. Designed to be cheap — called every `metric_interval_ticks`.

    Temp follows a first-order lag toward load-driven target plus pressure-driven drift.
    Counter rates spike when pressure is high — that's what makes them leading indicators.
    """
    rows: list[dict] = []
    for node in fleet.nodes:
        for gpu in node.gpus:
            gid = gpu.gpu_id
            load = load_by_gpu.get(gid, 0.0)
            st = bus.get(gid)

            target_temp = AMBIENT_C + 35.0 + 35.0 * load + 6.0 * st.ecc_pressure
            st.temp_c += 0.15 * (target_temp - st.temp_c) + rng.gauss(0, 1.2)

            power_w = TDP_W * (IDLE_FRACTION + (1 - IDLE_FRACTION) * load)
            power_w += 30 * st.nvlink_pressure + 20 * st.pcie_pressure
            power_w += rng.gauss(0, 8)

            sm_util = max(0.0, min(1.0, load + rng.gauss(0, 0.04)))

            # background counter drift — small Poisson-like increment when no event
            if rng.random() < 0.002 + 0.05 * st.ecc_pressure:
                st.ecc_corrected += 1
            if rng.random() < 0.001 + 0.08 * st.nvlink_pressure:
                st.nvlink_crc += rng.randint(1, 4)
            if rng.random() < 0.0008 + 0.06 * st.pcie_pressure:
                st.pcie_aer += 1

            rows.append(
                {
                    "tick": tick,
                    "gpu_id": gid,
                    "node_id": node.node_id,
                    "rack_id": node.rack_id,
                    "zone_id": node.zone_id,
                    "gpu_temp_c": round(st.temp_c, 2),
                    "gpu_power_w": round(power_w, 1),
                    "gpu_sm_util": round(sm_util, 3),
                    "ecc_corrected_total": st.ecc_corrected,
                    "nvlink_crc_total": st.nvlink_crc,
                    "pcie_aer_total": st.pcie_aer,
                }
            )
    return rows
