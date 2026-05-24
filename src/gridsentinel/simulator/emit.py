"""Run the simulation and write telemetry + tickets to data/synthetic/."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from gridsentinel.simulator import failure_models as fm
from gridsentinel.simulator.metrics import MetricsBus, sample_metrics
from gridsentinel.simulator.topology import build_fleet
from gridsentinel.simulator.workload import utilization_per_gpu

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"


def run(days: int = 14, seed: int = 42, ticks_per_hour: int = 12) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fleet = build_fleet()
    gpu_state: dict[str, dict] = {}
    ecc_state: dict[str, int] = {}
    nvlink_state: dict[str, int] = {}
    rack_thermal_stress: dict[str, float] = {}
    stress_decay = 0.95  # per-tick decay → half-life ~14 ticks (~70 min)

    # quick lookup: gpu_id → rack_id (for thermal stress accounting)
    gpu_to_rack: dict[str, str] = {
        gpu.gpu_id: node.rack_id for node in fleet.nodes for gpu in node.gpus
    }

    total_ticks = days * 24 * ticks_per_hour
    events: list[fm.FailureEvent] = []
    util_rows: list[dict] = []
    metric_rows: list[dict] = []
    metrics_bus = MetricsBus()
    metric_interval_ticks = max(1, ticks_per_hour // 12)  # ~5-minute cadence

    for tick in range(total_ticks):
        util = utilization_per_gpu(fleet, tick, rng, ticks_per_hour=ticks_per_hour)
        events.extend(fm.cooling_failure(fleet, tick, rng, ticks_per_hour=ticks_per_hour))
        thermal_events = fm.thermal_throttle(fleet, tick, rng, gpu_state, util)
        events.extend(thermal_events)
        # accumulate thermal stress at the rack level (decay + new warns/criticals)
        for r in rack_thermal_stress:
            rack_thermal_stress[r] *= stress_decay
        for ev in thermal_events:
            if ev.severity in ("warn", "critical"):
                r = gpu_to_rack.get(ev.target, "")
                rack_thermal_stress[r] = rack_thermal_stress.get(r, 0.0) + 1.0

        events.extend(
            fm.ecc_uncorrectable(fleet, tick, rng, ecc_state=ecc_state, ticks_per_hour=ticks_per_hour)
        )
        events.extend(
            fm.nvlink_fault(
                fleet, tick, rng, nvlink_state=nvlink_state, ticks_per_hour=ticks_per_hour
            )
        )
        events.extend(fm.pcie_error(fleet, tick, rng, ticks_per_hour=ticks_per_hour))
        events.extend(
            fm.psu_trip(
                fleet, tick, rng,
                rack_thermal_stress=rack_thermal_stress,
                ticks_per_hour=ticks_per_hour,
            )
        )
        events.extend(fm.network_flap(fleet, tick, rng, ticks_per_hour=ticks_per_hour))
        events.extend(fm.power_event(fleet, tick, rng, ticks_per_hour=ticks_per_hour))

        # Push failure events into the metrics bus so counters spike.
        for ev in events[-200:]:  # only check the just-emitted batch
            if ev.tick != tick:
                continue
            if ev.kind == fm.FailureKind.ECC_UNCORRECTABLE and "/" in ev.target:
                metrics_bus.bump_ecc(ev.target, n=2 if not ev.benign else 1)
            elif ev.kind == fm.FailureKind.NVLINK_FAULT and ev.scope == "node":
                # NVLink lives on a pair of GPUs — bump first two
                for gpu_idx in (0, 1):
                    metrics_bus.bump_nvlink(f"{ev.target}/gpu{gpu_idx}")
            elif ev.kind == fm.FailureKind.PCIE_ERROR and ev.scope == "node":
                for gpu_idx in (0, 1):
                    metrics_bus.bump_pcie(f"{ev.target}/gpu{gpu_idx}")

        # Sample continuous metrics at the configured cadence.
        if tick % metric_interval_ticks == 0:
            metric_rows.extend(sample_metrics(fleet, tick, rng, metrics_bus, util))
        metrics_bus.decay()
        # sample utilization once per hour to keep file size manageable
        if tick % ticks_per_hour == 0:
            for gpu_id, u in util.items():
                util_rows.append({"tick": tick, "gpu_id": gpu_id, "util": u})

    pd.DataFrame([e.__dict__ for e in events]).to_parquet(DATA_DIR / "events.parquet")
    pd.DataFrame(util_rows).to_parquet(DATA_DIR / "utilization.parquet")
    if metric_rows:
        pd.DataFrame(metric_rows).to_parquet(DATA_DIR / "metrics.parquet")
    pd.DataFrame(
        [
            {
                "node_id": n.node_id,
                "rack_id": n.rack_id,
                "zone_id": n.zone_id,
                "feed_id": n.feed_id,
                "spine_id": n.spine_id,
            }
            for n in fleet.nodes
        ]
    ).to_parquet(DATA_DIR / "fleet.parquet")
    print(
        f"wrote {len(events)} events, {len(util_rows)} util rows, "
        f"{len(metric_rows):,} metric samples, "
        f"{len(fleet.nodes)} nodes to {DATA_DIR}"
    )


if __name__ == "__main__":
    run()
