import pandas as pd

from gridsentinel.kpi import sla
from gridsentinel.kpi.efficiency import idle_energy_waste_kwh, node_power_watts


def test_availability_basic():
    assert sla.availability(1000, 0) == 1.0
    assert sla.availability(1000, 100) == 0.9
    assert sla.availability(0, 0) == 0.0


def test_mtbf():
    assert sla.mtbf(1000, 10) == 100.0
    assert sla.mtbf(1000, 0) == float("inf")


def test_alert_compression():
    assert sla.alert_compression_ratio(1000, 100) == 10.0


def test_noise_rate():
    assert sla.noise_rate(70, 100) == 0.7
    assert sla.noise_rate(0, 0) == 0.0


def test_idle_energy_waste():
    util_df = pd.DataFrame({
        "tick": [0, 0, 1, 1],
        "gpu_id": ["a", "b", "a", "b"],
        "util": [0.0, 0.5, 0.02, 0.9],
    })
    # 3 idle samples (util < 0.05): two from tick 0 ("a", util 0.0 — wait, b is 0.5 so only a)
    # tick 0: a is idle. tick 1: a (0.02) is idle.   → 2 idle samples.
    kwh = idle_energy_waste_kwh(util_df, idle_threshold=0.05)
    assert kwh > 0


def test_node_power_scales_with_util():
    idle = node_power_watts([0.0] * 8)
    full = node_power_watts([1.0] * 8)
    assert full > idle
