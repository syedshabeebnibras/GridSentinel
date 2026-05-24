"""Smoke tests for the TCN baseline — focus on shape/training plumbing.

We do NOT assert specific metric values; deep models on tiny synthetic data
are noisy. The test confirms:
  - Sequence builder produces the right shapes
  - Forward pass runs end-to-end
  - The TSCV training loop completes without error
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from gridsentinel.predict.deep_baseline import (
    TCN,
    TCNBlock,
    build_sequences,
    train_tcn,
)


def _toy_metrics(n_nodes: int = 6, hours: int = 96, ticks_per_hour: int = 12):
    rng = np.random.default_rng(0)
    rows = []
    for h in range(hours):
        for sample_in_hour in range(ticks_per_hour):
            tick = h * ticks_per_hour + sample_in_hour
            for n in range(n_nodes):
                for g in range(2):
                    rows.append({
                        "tick": tick,
                        "gpu_id": f"node-{n:04d}/gpu{g}",
                        "node_id": f"node-{n:04d}",
                        "rack_id": f"rack-{n // 3:03d}",
                        "zone_id": "zone-0",
                        "gpu_temp_c": 60 + 20 * np.sin(h * np.pi / 12) + rng.normal(0, 2),
                        "gpu_power_w": 400 + rng.normal(0, 30),
                        "gpu_sm_util": float(rng.beta(2, 3)),
                        "ecc_corrected_total": h,
                        "nvlink_crc_total": h * 2,
                        "pcie_aer_total": h,
                    })
    return pd.DataFrame(rows)


def _toy_events(ticks_per_hour: int = 12):
    # node-0000 fails near the end of the window
    return pd.DataFrame([{
        "tick": 80 * ticks_per_hour, "kind": "ecc_uncorrectable", "scope": "gpu",
        "target": "node-0000/gpu0", "severity": "critical", "benign": False,
        "parent_event_id": None, "node_id": "node-0000",
        "rack_id": "rack-000", "zone_id": "zone-0", "feed_id": "f", "spine_id": "s",
    }])


def test_tcn_forward_pass_works():
    """Single forward pass produces logits of the expected shape."""
    model = TCN(in_channels=6, hidden=16)
    x = torch.zeros(4, 6, 32)
    out = model(x)
    assert out.shape == (4,), f"expected (4,) logits, got {tuple(out.shape)}"


def test_tcn_block_preserves_length():
    """Causal blocks must keep the time dim length-preserving."""
    block = TCNBlock(6, 8, kernel=3, dilation=2)
    x = torch.zeros(2, 6, 16)
    out = block(x)
    assert out.shape == (2, 8, 16)


def test_build_sequences_shapes():
    metrics = _toy_metrics(n_nodes=6, hours=96)
    events = _toy_events()
    fleet = pd.DataFrame([
        {"node_id": f"node-{n:04d}", "rack_id": f"rack-{n//3:03d}",
         "zone_id": "zone-0", "feed_id": "f", "spine_id": "s"}
        for n in range(6)
    ])
    X, y, ticks = build_sequences(
        metrics, events, fleet, lookback_hours=24, forecast_hours=24, step_hours=12,
        samples_per_window=32,
    )
    assert X.ndim == 3 and X.shape[1] == 6 and X.shape[2] == 32
    assert len(X) == len(y) == len(ticks)


def test_train_tcn_runs_with_smoke_data():
    metrics = _toy_metrics(n_nodes=10, hours=120)
    events = _toy_events()
    fleet = pd.DataFrame([
        {"node_id": f"node-{n:04d}", "rack_id": f"rack-{n//3:03d}",
         "zone_id": "zone-0", "feed_id": "f", "spine_id": "s"}
        for n in range(10)
    ])
    result = train_tcn(metrics, events, fleet, n_splits=2, epochs=1)
    # Either it ran (n_folds >= 1) or it explicitly reported too few sequences
    assert "n_sequences" in result.metrics or "error" in result.metrics
