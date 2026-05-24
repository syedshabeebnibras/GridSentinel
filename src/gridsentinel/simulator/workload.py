"""Workload patterns — drives GPU utilization per node over the simulation.

Shape:
  - baseline 0.45
  - **diurnal** sine wave (afternoon peak)
  - **weekly** dip on weekends
  - **slow growth** in baseline (0.5%/day) → makes capacity forecast meaningful
  - per-GPU gaussian noise

The growth term is what turns a flat fleet into a forecastable one: without
it, SARIMAX projects a flat line forever and "weeks to 85% load" is always ∞.
"""
from __future__ import annotations

import math
import random

from gridsentinel.simulator.topology import Fleet


def utilization_per_gpu(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    ticks_per_hour: int = 12,
) -> dict[str, float]:
    hour_of_day = (tick // ticks_per_hour) % 24
    day_of_run = tick // (ticks_per_hour * 24)
    day_of_week = day_of_run % 7

    diurnal = 0.25 * math.sin((hour_of_day - 6) * math.pi / 12)
    weekend_dip = -0.10 if day_of_week >= 5 else 0.0
    growth = 0.008 * day_of_run  # +0.8%/day baseline drift — makes forecast cross 85%

    util: dict[str, float] = {}
    for node in fleet.nodes:
        for gpu in node.gpus:
            base = 0.45 + diurnal + weekend_dip + growth + rng.gauss(0, 0.08)
            util[gpu.gpu_id] = max(0.0, min(1.0, base))
    return util
