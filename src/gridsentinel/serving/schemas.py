"""Pydantic schemas for the prediction API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    version: str
    model_version: str | None
    n_features: int | None
    trained_at: str | None
    uptime_seconds: float


class VersionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    n_training_rows: int
    n_features: int
    feature_names: list[str]
    metrics: dict[str, Any]
    survival_metrics: dict[str, Any] | None = None
    data_hash: str
    trained_at: str


class PredictRequest(BaseModel):
    """Per-node feature vector for one prediction.

    Two ways to call it:
      1. `features` dict — keys = feature names, values = numbers. Missing
         features default to 0. Extra features are ignored.
      2. `node_id` — score the latest cached window for that node id (only
         available if the server was started with `--cache-windows`).
    """
    node_id: str | None = None
    features: dict[str, float] | None = None


class AtRiskNode(BaseModel):
    node_id: str
    rack_id: str | None = None
    zone_id: str | None = None
    failure_risk_24h: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    node_id: str | None
    failure_risk_24h: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    timestamp: datetime


class ExplainRequest(BaseModel):
    node_id: str | None = None
    features: dict[str, float] | None = None
    top_n: int = Field(default=6, ge=1, le=30)


class ExplainEntry(BaseModel):
    feature: str
    value: float
    shap: float


class ExplainResponse(BaseModel):
    node_id: str | None
    failure_risk_24h: float
    top_features: list[ExplainEntry]


class AtRiskResponse(BaseModel):
    n: int
    nodes: list[AtRiskNode]
    generated_at: datetime
