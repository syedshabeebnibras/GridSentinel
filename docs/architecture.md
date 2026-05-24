# GridSentinel architecture

```
              ┌────────────────────┐
              │   simulator        │  500 nodes × 8 GPUs, correlated failures
              │  (topology +       │
              │  failure_models +  │
              │  workload + emit)  │
              └────────┬───────────┘
                       │ parquet
                       ▼
              ┌────────────────────┐
              │   ingest           │  telemetry / tickets
              └────────┬───────────┘
                       │ tidy frames
              ┌────────┴───────────┐
              │                    │
              ▼                    ▼
       ┌────────────┐       ┌────────────┐
       │ correlation│       │   kpi      │  availability, MTTR, MTBF,
       │ dedupe →   │       │  sla +     │  perf/W, idle waste, PUE
       │ correlate  │       │  efficiency│
       └─────┬──────┘       └─────┬──────┘
             │ incidents          │ kpis
             ▼                    │
       ┌────────────┐             │
       │  rootcause │             │
       │ features → │             │
       │ cluster    │             │
       └─────┬──────┘             │
             │ clusters           │
             ▼                    │
       ┌────────────┐             │
       │  forecast  │             │
       │  utilizn → │             │
       │  capacity  │             │
       └─────┬──────┘             │
             │ horizon            │
             ▼                    ▼
              ┌────────────────────┐
              │   reporting        │  weekly LLM-authored narrative
              │   render           │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │   dashboard        │  Streamlit (or Power BI / Tableau export)
              └────────────────────┘
```

## Build sequence

1. **simulator** — get realistic correlated events flowing into parquet.
   - Implement `failure_models.thermal_throttle` first (TODO marked).
   - Then ECC, NVLink, PCIe (stateless).
   - Then PSU, network_flap, power_event (correlated to rack / spine / feed).
2. **ingest + kpi** — get availability/MTTR/perf-per-watt computing on the synthetic data.
3. **dashboard** — first end-to-end deliverable. Already resume-worthy.
4. **correlation** — dedupe + correlate. Validates the 10× compression KPI.
5. **rootcause** — featurize + HDBSCAN; produce top-N recurring families.
6. **forecast** — SARIMAX on daily GPU-hours; weeks-to-85%-load.
7. **reporting** — wire KPIs + clusters + forecast into the weekly markdown.
