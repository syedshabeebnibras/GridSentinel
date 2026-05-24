# GridSentinel

A closed-loop GPU data-center operations intelligence system. Simulates a 4,000-GPU
HPC fleet, correlates noisy telemetry into root-cause incidents, forecasts
capacity headroom, surfaces idle-energy waste, and auto-generates a weekly
analyst-voice report.

## Pipeline

```
simulator → ingest → dedupe → correlate → root-cause cluster
                                              ↓
                              forecast       energy       predictive-maintenance ML
                                              ↓
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
              Streamlit dashboard      Weekly LLM report      Power BI + Tableau exports
                                       (data/reports/)        (parquet + CSV + DAX/Calc)
```

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Generate 14 days of synthetic telemetry + tickets
python -m gridsentinel.simulator.emit

# 2. Live dashboard
streamlit run src/gridsentinel/dashboard/app.py

# 3. Export the same data to Power BI Desktop star schema
python -m gridsentinel.powerbi.export
# → opens at data/powerbi/ — see README there for Power BI Desktop setup

# 4. Also export to Tableau Public (CSV only) + build the .twbx workbook
python -m gridsentinel.tableau.export
python -m gridsentinel.bi.twbx
# → data/tableau/GridSentinel.twbx — open in Tableau Public Desktop on macOS

# 4b. Build the Power BI template (.pbit) + fallback .pq script
python -m gridsentinel.bi.pbit
# → data/powerbi/GridSentinel.pbit (Windows Power BI Desktop)
# → data/powerbi/GridSentinel.pq   (paste into Advanced Editor as a fallback)

# 5. Train + score the predictive-maintenance model
python -m gridsentinel.predict.score
# → ROC AUC + Precision@K, top 10 at-risk nodes for next 24h
# → data/predict/feature_importance.csv, top_at_risk.csv, metrics.csv

# 6. Generate the weekly auto-written report
python -m gridsentinel.reporting.render
# → data/reports/weekly_YYYY-WNN.md
# (set ANTHROPIC_API_KEY in .env to get LLM-narrative paragraphs;
#  otherwise template-rendered)
```

## What's modeled

**Topology:** 500 nodes × 8 H100-class GPUs, 100 racks, 5 cooling zones, 2 power feeds/zone, 2 network spines.

**Failure modes** (each with correct correlation key for the topology):

| Mode | Scope | Cascades to |
|---|---|---|
| Thermal throttling | GPU | — (stateful temperature walk) |
| ECC uncorrectable | GPU | — |
| NVLink fault | Node | — |
| PCIe error | Node | — |
| Cooling failure | Zone | all 100 nodes in zone |
| PSU trip | Rack | all 5 nodes on rack |
| Network flap | Spine | ~30% of nodes on spine |
| Power event | Feed | all racks on feed |

**Workload:** diurnal sine + weekly weekend dip + 0.8%/day baseline growth (makes capacity forecast meaningful).

**Stateful precursor patterns (for predictive maintenance):**

- ECC uncorrectable is preceded by corrected ECC errors on the same GPU (escalation_threshold=1, 95% critical after threshold)
- NVLink critical fault is preceded by 2+ NVLink warns on the same node (70% critical after threshold)
- PSU trip hazard scales with rolling rack thermal stress (sustained heat → shorter PSU life)

## KPIs computed

- **SLA:** availability, MTBF, MTTR, MTTD, incident rate, alert compression ratio, noise rate
- **Energy:** avg fleet draw kW, total kWh, idle energy waste kWh + $, performance per Watt (TFLOPS/W), PUE, renewable match
- **Capacity:** current headroom %, weeks to 70% load (trend-on-daily-means + diurnal decomposition)
- **Cluster quality:** silhouette score
- **Predictive maintenance (v2 — industry-grade pipeline):**
  - **Continuous DCGM-style telemetry**: per-5min GPU temp, power, SM util, ECC/NVLink/PCIe counters
  - **Hybrid features**: 30+ rolling stats on continuous streams (mean/std/slope/autocorr/spike count/volatility regime) **+** per-(kind, severity) event counts
  - **Calibrated classifier** (HistGradientBoosting + isotonic regression) — proper probabilities
  - **Cox proportional hazards survival model** (lifelines) — time-to-next-failure with C-index
  - **IsolationForest anomaly detector** — defense in depth for unknown failure modes
  - **Rolling-origin TimeSeriesSplit** (5 folds) — reports mean ± std for every metric
  - **SHAP explanations** — global feature importance + per-prediction "why this node?"
  - **MLflow experiment tracking** at `mlruns/`
  - **Versioned model registry** at `data/models/v{N}/` with metadata.json + drift baseline
  - **PSI drift detector** — Population Stability Index per feature against training baseline

## Stack

Python 3.13 · pandas · NumPy · scikit-learn (HDBSCAN, HistGradientBoosting + CalibratedClassifierCV + IsolationForest + TimeSeriesSplit) · statsmodels · **lifelines** (Cox PH) · **PyTorch** (TCN deep baseline) · **SHAP** · **MLflow** · joblib · Streamlit · **FastAPI** (online prediction service) · Power BI star-schema export (parquet + DAX) · Tableau Public export (CSV + Calculated Fields) · DCGM/Prometheus/OpenTelemetry ingest adapters · Docker (deployable image) · optional Anthropic SDK (Claude Haiku 4.5) for LLM-narrative reports.

## Real-data ingestion paths

Three adapters convert real DC telemetry into the schema GridSentinel's
pipeline already consumes:

```python
from gridsentinel.ingest import dcgm, prometheus, otel

# 1. DCGM CSV export
df = dcgm.from_dcgm("dcgm_export.csv")

# 2. Prometheus query_range JSON snapshot
df = prometheus.from_file("prom_snapshot.json")

# 3. Live Prometheus / Mimir / VictoriaMetrics HTTP query
df = prometheus.from_http(base_url="http://prometheus.internal:9090")

# 4. OpenTelemetry OTLP/JSON
df = otel.from_file("otlp.json")
```

All four return the same `data/synthetic/metrics.parquet` schema, so the
rest of the pipeline (features, model, dashboard) is indifferent to the
source.

## Online prediction service

```bash
uvicorn gridsentinel.serving.app:app --port 8080
# → http://localhost:8080/docs for OpenAPI / Swagger UI
```

Endpoints:
- `GET /health`     — liveness + registered model version
- `GET /version`    — full model metadata (features, metrics, data_hash)
- `POST /predict`   — score a single feature vector or cached node
- `POST /explain`   — per-prediction SHAP attribution
- `GET /at-risk`    — top-N at-risk leaderboard

Containerized via `Dockerfile`. Build with `docker build -t gridsentinel-api .`
