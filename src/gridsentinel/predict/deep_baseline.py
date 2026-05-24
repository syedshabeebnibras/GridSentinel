"""Temporal Convolutional Network baseline — the deep-learning yardstick.

A TCN consumes the raw continuous metric sequences per node-window (no
hand-engineered features) and produces a failure-probability. We compare its
TSCV metrics directly against the calibrated gradient-boosting model.

Honest expectation: on tabular telemetry with rolling stats already engineered,
GBM usually beats DL. The point of this baseline is not to win — it is to
demonstrate that we benchmarked against the obvious deep alternative and
chose GBM for empirically defensible reasons (Occam plus tabular wins).

Architecture:
  - 1D causal dilated convolutions (Bai/Kolter/Koltun 2018 TCN)
  - 3 residual blocks with dilation 1, 2, 4 (receptive field ~14 samples)
  - Global average pooling, linear head, sigmoid

The TCN expects input shape (batch, channels=6, length=L) where the 6
channels are gpu_temp_c, gpu_power_w, gpu_sm_util plus three counter
rates (ecc, nvlink, pcie) computed from monotonic counters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_CHANNELS = (
    "gpu_temp_c", "gpu_power_w", "gpu_sm_util",
    "ecc_corrected_rate", "nvlink_crc_rate", "pcie_aer_rate",
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Model -----------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """Dilated causal 1D conv that left-pads to keep the model strictly causal."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.c2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.c1(x))
        h = self.drop(h)
        h = F.gelu(self.c2(h))
        h = self.norm(h)
        return F.gelu(h + self.skip(x))


class TCN(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 32, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.b1 = TCNBlock(in_channels, hidden, kernel, dilation=1, dropout=dropout)
        self.b2 = TCNBlock(hidden, hidden, kernel, dilation=2, dropout=dropout)
        self.b3 = TCNBlock(hidden, hidden, kernel, dilation=4, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.b3(self.b2(self.b1(x)))
        pooled = h.mean(dim=-1)
        return self.head(pooled).squeeze(-1)


# Data shaping ---------------------------------------------------------------
def build_sequences(
    metrics_df: pd.DataFrame,
    enriched_events: pd.DataFrame,
    fleet: pd.DataFrame,
    ticks_per_hour: int = 12,
    lookback_hours: int = 48,
    forecast_hours: int = 24,
    step_hours: int = 6,
    samples_per_window: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, 6, samples_per_window), y, window_end_ticks."""
    if metrics_df.empty:
        return (
            np.empty((0, 6, samples_per_window)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    m = metrics_df.sort_values(["gpu_id", "tick"]).copy()
    m["ecc_corrected_rate"] = (
        m.groupby("gpu_id")["ecc_corrected_total"].diff().fillna(0).clip(lower=0)
    )
    m["nvlink_crc_rate"] = (
        m.groupby("gpu_id")["nvlink_crc_total"].diff().fillna(0).clip(lower=0)
    )
    m["pcie_aer_rate"] = (
        m.groupby("gpu_id")["pcie_aer_total"].diff().fillna(0).clip(lower=0)
    )

    agg = (
        m.groupby(["node_id", "tick"]).agg(
            gpu_temp_c=("gpu_temp_c", "mean"),
            gpu_power_w=("gpu_power_w", "mean"),
            gpu_sm_util=("gpu_sm_util", "mean"),
            ecc_corrected_rate=("ecc_corrected_rate", "sum"),
            nvlink_crc_rate=("nvlink_crc_rate", "sum"),
            pcie_aer_rate=("pcie_aer_rate", "sum"),
        )
        .reset_index()
    )

    max_tick = int(max(metrics_df["tick"].max(), enriched_events["tick"].max()))
    lookback = lookback_hours * ticks_per_hour
    horizon = forecast_hours * ticks_per_hour
    step = step_hours * ticks_per_hour
    component_kinds = ["ecc_uncorrectable", "nvlink_fault", "pcie_error"]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    win_ticks: list[int] = []

    node_ids = sorted(agg["node_id"].unique())
    agg_indexed = {nid: g.sort_values("tick").reset_index(drop=True) for nid, g in agg.groupby("node_id")}

    for window_end in range(lookback, max_tick - horizon + 1, step):
        win_start = window_end - lookback
        target_end = window_end + horizon
        crit = (
            enriched_events.loc[
                (enriched_events["tick"] >= window_end)
                & (enriched_events["tick"] < target_end)
                & (enriched_events["severity"] == "critical")
                & (enriched_events["kind"].isin(component_kinds))
            ]
            .groupby("node_id")
            .size()
        )
        for nid in node_ids:
            ng = agg_indexed.get(nid)
            if ng is None:
                continue
            win = ng[(ng["tick"] >= win_start) & (ng["tick"] < window_end)]
            if len(win) < 4:
                continue
            idx = np.linspace(0, len(win) - 1, samples_per_window).round().astype(int)
            seq = win[list(_CHANNELS)].iloc[idx].to_numpy(dtype=np.float32).T
            xs.append(seq)
            ys.append(int(crit.get(nid, 0) > 0))
            win_ticks.append(window_end)

    if not xs:
        return (
            np.empty((0, 6, samples_per_window)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )
    return np.stack(xs), np.array(ys, dtype=np.int64), np.array(win_ticks, dtype=np.int64)


# Train / evaluate -----------------------------------------------------------
@dataclass
class TCNResult:
    metrics: dict[str, float]
    cv_metrics: dict[str, dict[str, float]]
    n_params: int
    device: str


def _fit_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 8, batch_size: int = 256, lr: float = 1e-3,
) -> dict[str, float]:
    device = _device()
    model = TCN(in_channels=X_train.shape[1]).to(device)

    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # switch to inference mode (PyTorch nn.Module.eval — disables dropout/BN updates)
    model.train(mode=False)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(device))
        proba = torch.sigmoid(logits).cpu().numpy()

    out: dict[str, float] = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_pos_rate": float(y_test.mean()),
    }
    if len(np.unique(y_test)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_test, proba))
        out["pr_auc"] = float(average_precision_score(y_test, proba))
        out["brier"] = float(brier_score_loss(y_test, proba))
        for k in (10, 25, 50):
            if len(proba) >= k:
                top_k_idx = np.argsort(proba)[-k:]
                out[f"precision_at_{k}"] = float(y_test[top_k_idx].mean())
                out[f"lift_at_{k}"] = out[f"precision_at_{k}"] / max(out["test_pos_rate"], 1e-6)
    return out


def train_tcn(
    metrics_df: pd.DataFrame,
    enriched_events: pd.DataFrame,
    fleet: pd.DataFrame,
    n_splits: int = 3,
    epochs: int = 8,
) -> TCNResult:
    """Train + evaluate the TCN with rolling-origin TSCV. Same fold structure
    as the GBM baseline so metrics are directly comparable."""
    X, y, ticks = build_sequences(metrics_df, enriched_events, fleet)
    if len(X) < 100:
        return TCNResult(
            metrics={"error": "not enough sequences", "n": int(len(X))},
            cv_metrics={}, n_params=0, device=str(_device()),
        )

    order = np.argsort(ticks)
    X, y, ticks = X[order], y[order], ticks[order]

    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    for fold, (tr, te) in enumerate(tss.split(X)):
        if len(np.unique(y[te])) < 2:
            continue
        nC, L = X.shape[1], X.shape[2]
        scaler = StandardScaler()
        train_flat = X[tr].transpose(0, 2, 1).reshape(-1, nC)
        scaler.fit(train_flat)

        def _scale(arr: np.ndarray) -> np.ndarray:
            flat = arr.transpose(0, 2, 1).reshape(-1, nC)
            return scaler.transform(flat).reshape(arr.shape[0], L, nC).transpose(0, 2, 1)

        Xtr_s = _scale(X[tr]).astype(np.float32)
        Xte_s = _scale(X[te]).astype(np.float32)
        m = _fit_fold(Xtr_s, y[tr], Xte_s, y[te], epochs=epochs)
        m["fold"] = fold
        fold_metrics.append(m)

    agg: dict[str, dict[str, float]] = {}
    if fold_metrics:
        for key in fold_metrics[0]:
            if key == "fold":
                continue
            vals = [m[key] for m in fold_metrics if key in m]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    sample_model = TCN(in_channels=X.shape[1])
    n_params = sum(p.numel() for p in sample_model.parameters())

    headline = {
        "n_sequences": int(len(X)),
        "n_folds": len(fold_metrics),
        "device": str(_device()),
        "roc_auc_mean": agg.get("roc_auc", {}).get("mean", float("nan")),
        "roc_auc_std": agg.get("roc_auc", {}).get("std", float("nan")),
        "pr_auc_mean": agg.get("pr_auc", {}).get("mean", float("nan")),
        "brier_mean": agg.get("brier", {}).get("mean", float("nan")),
        "precision_at_10_mean": agg.get("precision_at_10", {}).get("mean", float("nan")),
        "lift_at_10_mean": agg.get("lift_at_10", {}).get("mean", float("nan")),
    }
    return TCNResult(metrics=headline, cv_metrics=agg, n_params=n_params, device=str(_device()))
