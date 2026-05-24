"""FastAPI online prediction service.

Endpoints:
  - GET  /health         liveness + model status
  - GET  /version        registered model metadata
  - POST /predict        score a single feature vector
  - POST /explain        per-prediction SHAP attribution
  - GET  /at-risk?n=10   leaderboard from the latest cached scoring window
  - GET  /docs           OpenAPI / Swagger UI (auto-generated)

The model registry is loaded once at startup. Reload by restarting the
process — keeps the request path zero-IO. For hot-reload in production wire
this to a registry watcher (out of scope here).

Run locally:
    uvicorn gridsentinel.serving.app:app --reload --port 8080
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gridsentinel import __version__ as pkg_version
from gridsentinel.correlation.correlate import correlate
from gridsentinel.correlation.dedupe import dedupe
from gridsentinel.ingest.telemetry import enrich_with_topology
from gridsentinel.predict.calibrated import shap_explanation_for_node
from gridsentinel.predict.registry import load_latest
from gridsentinel.predict.ts_features import (
    build_timeseries_dataset,
    ts_feature_columns,
)
from gridsentinel.serving.schemas import (
    AtRiskNode,
    AtRiskResponse,
    ExplainEntry,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)


# Module-level state populated at startup
_state: dict[str, Any] = {
    "registry": None,
    "started_at": time.time(),
    "latest_windows": pd.DataFrame(),
    "feature_names": [],
}


_SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}


def _build_summary(events_path: Path, fleet_path: Path) -> dict[str, Any]:
    """Pre-compute all dashboard panel data once at startup.

    Returns a JSON-serialisable dict the /summary endpoint hands to the web
    showcase. Cheap to recompute (a few seconds on ~200k events) but we cache
    it so /summary itself is O(1).
    """
    if not (events_path.exists() and fleet_path.exists()):
        return {}

    try:
        events = pd.read_parquet(events_path)
        fleet = pd.read_parquet(fleet_path)
    except Exception as e:
        print(f"warning: /summary build failed reading inputs: {e}")
        return {}

    try:
        enriched = enrich_with_topology(events, fleet)
        deduped = dedupe(enriched, window_ticks=5)
        incidents = correlate(deduped, time_window_ticks=15)
    except Exception as e:
        print(f"warning: /summary correlation failed: {e}")
        return {}

    event_kinds = (
        events.groupby("kind").size().sort_values(ascending=False).to_dict()
    )
    severity_raw = events["severity"].value_counts().to_dict()
    severity_incidents = incidents["severity_max"].value_counts().to_dict()

    top_incidents = (
        incidents.nlargest(10, "member_count")[
            ["incident_id", "root_kind", "scope", "member_count",
             "severity_max", "any_benign"]
        ]
        .to_dict("records")
    )

    # Mean util series — derive from incidents' tick spread to keep the API
    # slim (no need to load 12 MB of utilization.parquet).
    tick_min = int(events["tick"].min())
    tick_max = int(events["tick"].max())
    bins = 48
    edges = list(range(tick_min, tick_max + 1, max(1, (tick_max - tick_min) // bins)))
    # Severity-weighted activity per tick bucket — proxy for "how busy was the fleet"
    activity_per_tick = events.assign(
        weight=events["severity"].map(_SEVERITY_RANK).fillna(0) + 1
    ).groupby(pd.cut(events["tick"], bins=edges), observed=True)["weight"].sum()
    util_series = [
        {"tick": int(interval.mid), "activity": int(val)}
        for interval, val in activity_per_tick.items()
        if pd.notna(interval)
    ]

    return {
        "event_kinds": [{"kind": k, "count": int(v)} for k, v in event_kinds.items()],
        "severity": {
            "raw": {k: int(severity_raw.get(k, 0)) for k in ["info", "warn", "critical"]},
            "incidents": {
                k: int(severity_incidents.get(k, 0)) for k in ["info", "warn", "critical"]
            },
        },
        "top_incidents": [
            {
                "incident_id": str(r["incident_id"])[:80],
                "root_kind": str(r["root_kind"]),
                "scope": str(r["scope"]),
                "member_count": int(r["member_count"]),
                "severity_max": str(r["severity_max"]),
                "any_benign": bool(r["any_benign"]),
            }
            for r in top_incidents
        ],
        "activity_series": util_series,
        "totals": {
            "raw_events": int(len(events)),
            "incidents": int(len(incidents)),
            "deduped": int(len(deduped)),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model registry + cache the latest window for at-risk queries."""
    registry = load_latest()
    _state["registry"] = registry
    if registry is None:
        yield
        return

    _state["feature_names"] = list(registry["calibrated"].feature_names)

    # Pre-compute the most recent (node, window) feature snapshot so
    # /at-risk and /predict-by-node are O(1) lookups instead of recomputing.
    # Preferred path: read the persisted parquet that `predict.score` writes
    # (small, ~50 KB). Fallback: rebuild from raw 16M-row metrics if present.
    repo_root = Path(__file__).resolve().parents[3]
    persisted = repo_root / "data" / "predict" / "latest_window.parquet"
    data_dir = repo_root / "data" / "synthetic"
    metrics_path = data_dir / "metrics.parquet"
    events_path = data_dir / "events.parquet"
    fleet_path = data_dir / "fleet.parquet"

    if persisted.exists():
        try:
            latest = pd.read_parquet(persisted)
            _state["latest_windows"] = latest.set_index("node_id")
        except Exception as e:
            print(f"warning: failed to load persisted latest_window: {e}")
    elif metrics_path.exists() and events_path.exists() and fleet_path.exists():
        try:
            metrics = pd.read_parquet(metrics_path)
            events = pd.read_parquet(events_path)
            fleet = pd.read_parquet(fleet_path)
            enriched = enrich_with_topology(events, fleet)
            ds = build_timeseries_dataset(metrics, enriched, fleet)
            if not ds.empty:
                latest = ds[ds["window_end_tick"] == ds["window_end_tick"].max()]
                _state["latest_windows"] = latest.set_index("node_id")
        except Exception as e:
            # serving should not die if the snapshot is missing/corrupt
            print(f"warning: failed to build latest-window cache: {e}")

    # Pre-compute the dashboard summary once at startup — all panel data
    # in one cheap blob the /summary endpoint serves to the web showcase.
    _state["summary"] = _build_summary(events_path, fleet_path)

    yield


app = FastAPI(
    title="GridSentinel Predictive Maintenance API",
    description=(
        "Online scoring service for 24h GPU-node failure risk. "
        "Combines a calibrated gradient-boosting classifier, Cox PH survival "
        "model, and IsolationForest anomaly detector, all loaded from a "
        "versioned model registry."
    ),
    version=pkg_version,
    lifespan=lifespan,
)

# CORS — allow the Vercel showcase site (and any custom domain) to fetch
# live /at-risk and /version data. Permissive in dev; tighten allow_origins
# in production if you add authenticated endpoints.
_default_origins = [
    "https://gridsentinel.vercel.app",
    "https://gridsentinel-7k47i21iw-syedshabeebnibras-projects.vercel.app",
    "http://localhost:3000",
    "http://localhost:8501",
]
_extra = os.environ.get("CORS_EXTRA_ORIGINS", "").split(",")
_origins = _default_origins + [o.strip() for o in _extra if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Cover every Vercel preview URL on this project (-* subdomains)
    allow_origin_regex=r"https://gridsentinel-[\w-]+\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


def _require_model():
    if _state["registry"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "no model registered. Run "
                "`python -m gridsentinel.predict.score` to train and register, "
                "then restart this service."
            ),
        )
    return _state["registry"]


def _features_to_dataframe(features: dict[str, float], cols: list[str]) -> pd.DataFrame:
    row = {c: float(features.get(c, 0.0)) for c in cols}
    return pd.DataFrame([row])


def _features_for_node(node_id: str) -> pd.DataFrame | None:
    latest: pd.DataFrame = _state["latest_windows"]
    if latest is None or latest.empty or node_id not in latest.index:
        return None
    row = latest.loc[[node_id]]
    return row[_state["feature_names"]]


@app.get("/health", response_model=HealthResponse)
async def health():
    registry = _state["registry"]
    meta = registry["metadata"] if registry else None
    return HealthResponse(
        status="ok" if registry else "no_model",
        version=pkg_version,
        model_version=registry["version"] if registry else None,
        n_features=meta["n_features"] if meta else None,
        trained_at=meta["created_at"] if meta else None,
        uptime_seconds=time.time() - _state["started_at"],
    )


@app.get("/version", response_model=VersionResponse)
async def version():
    registry = _require_model()
    meta = registry["metadata"]
    return VersionResponse(
        model_version=registry["version"],
        n_training_rows=meta["n_training_rows"],
        n_features=meta["n_features"],
        feature_names=meta["feature_names"],
        metrics=meta["metrics"],
        survival_metrics=meta.get("survival_metrics"),
        data_hash=meta["data_hash"],
        trained_at=meta["created_at"],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    registry = _require_model()
    cal = registry["calibrated"]
    anom = registry["anomaly"]
    cols = cal.feature_names

    if req.features is not None:
        X = _features_to_dataframe(req.features, cols)
    elif req.node_id is not None:
        X = _features_for_node(req.node_id)
        if X is None:
            raise HTTPException(
                status_code=404,
                detail=f"node {req.node_id!r} not in cached latest window — "
                       "either supply `features` directly or rebuild the cache.",
            )
    else:
        raise HTTPException(status_code=400, detail="provide either `features` or `node_id`")

    proba = float(cal.predict_proba(X)[0])
    anomaly_score = float(anom.score(X)[0])
    return PredictResponse(
        node_id=req.node_id,
        failure_risk_24h=proba,
        anomaly_score=anomaly_score,
        timestamp=datetime.utcnow(),
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    registry = _require_model()
    cal = registry["calibrated"]
    cols = cal.feature_names

    if req.features is not None:
        X = _features_to_dataframe(req.features, cols)
    elif req.node_id is not None:
        X = _features_for_node(req.node_id)
        if X is None:
            raise HTTPException(status_code=404, detail=f"node {req.node_id!r} unknown")
    else:
        raise HTTPException(status_code=400, detail="provide either `features` or `node_id`")

    risk = float(cal.predict_proba(X)[0])
    shap_df = shap_explanation_for_node(cal, X, top_n=req.top_n)
    entries = [
        ExplainEntry(feature=str(r["feature"]), value=float(r["value"]), shap=float(r["shap"]))
        for _, r in shap_df.iterrows()
    ]
    return ExplainResponse(node_id=req.node_id, failure_risk_24h=risk, top_features=entries)


@app.get("/summary")
async def summary():
    """Pre-computed dashboard panel data: event kinds, severity mix, top
    incidents, activity timeseries. Cached at startup; restart the service
    to refresh after a new simulation."""
    if _state["registry"] is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    payload = _state.get("summary") or {}
    return payload


@app.get("/at-risk", response_model=AtRiskResponse)
async def at_risk(n: int = 10):
    registry = _require_model()
    latest: pd.DataFrame = _state["latest_windows"]
    if latest is None or latest.empty:
        return AtRiskResponse(n=0, nodes=[], generated_at=datetime.utcnow())

    cal = registry["calibrated"]
    anom = registry["anomaly"]
    X = latest[cal.feature_names]
    proba = cal.predict_proba(X)
    anomaly = anom.score(X)
    df = latest.reset_index()[["node_id", "rack_id", "zone_id"]].copy()
    df["failure_risk_24h"] = proba
    df["anomaly_score"] = anomaly
    top = df.sort_values("failure_risk_24h", ascending=False).head(n)
    nodes = [
        AtRiskNode(
            node_id=str(r["node_id"]),
            rack_id=None if pd.isna(r["rack_id"]) else str(r["rack_id"]),
            zone_id=None if pd.isna(r["zone_id"]) else str(r["zone_id"]),
            failure_risk_24h=float(r["failure_risk_24h"]),
            anomaly_score=float(r["anomaly_score"]),
        )
        for _, r in top.iterrows()
    ]
    return AtRiskResponse(n=len(nodes), nodes=nodes, generated_at=datetime.utcnow())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """Don't leak stack traces to the client; surface a clean 500."""
    return JSONResponse(status_code=500, content={"detail": f"server error: {type(exc).__name__}"})
