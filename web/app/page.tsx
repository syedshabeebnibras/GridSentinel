import fs from "node:fs/promises";
import path from "node:path";
import Image from "next/image";

type Snapshot = {
  generated_at: string;
  model_version: string;
  data_hash: string;
  kpis: {
    raw_events: number;
    incidents: number;
    compression_x: number;
    noise_rate: number;
    critical_incidents: number;
    fleet_kw_avg: number;
    total_kwh: number;
    idle_kwh: number;
    idle_waste_dollars: number;
    perf_per_watt_tflops: number;
  };
  pdm_metrics: {
    roc_auc_mean: number;
    roc_auc_std: number;
    pr_auc_mean: number;
    brier_mean: number;
    precision_at_10_mean: number;
    lift_at_10_mean: number;
    cox_c_index: number | null;
    n_features: number;
    n_folds: number;
    n_training_rows: number;
  };
  top_at_risk: Array<{
    node_id: string;
    rack_id: string;
    zone_id: string;
    failure_risk_24h: number;
    anomaly_score: number;
  }>;
  top_features: Array<{ feature: string; mean_abs_shap: number }>;
};

async function loadSnapshot(): Promise<Snapshot> {
  const p = path.join(process.cwd(), "public", "data", "snapshot.json");
  const raw = await fs.readFile(p, "utf8");
  return JSON.parse(raw);
}

export const revalidate = 3600;

const fmt = new Intl.NumberFormat("en-US");
const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default async function Page() {
  const snap = await loadSnapshot();
  const k = snap.kpis;
  const m = snap.pdm_metrics;

  return (
    <main className="min-h-screen">
      {/* Glow header */}
      <div className="glow border-b border-line">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <div className="flex items-center gap-3 text-mute text-sm font-mono mb-6">
            <span className="inline-block w-2 h-2 rounded-full bg-accent shadow-[0_0_12px_rgba(94,234,212,0.7)]" />
            <span>model {snap.model_version} · trained {new Date(snap.generated_at).toUTCString()}</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
            GridSentinel
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-mute leading-relaxed">
            A closed-loop GPU data-center operations + predictive-maintenance platform.
            Simulates a 4,000-GPU fleet, compresses noisy telemetry into root-cause
            incident clusters, forecasts capacity, predicts node failure 24 hours
            ahead, and auto-writes the weekly ops report.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="https://github.com/syedshabeebnibras/GridSentinel"
              className="px-4 py-2 rounded-md bg-accent text-bg font-medium hover:opacity-90 transition"
            >
              View on GitHub →
            </a>
            <a
              href="#pdm"
              className="px-4 py-2 rounded-md border border-line hover:border-accent transition"
            >
              Skip to the ML pipeline
            </a>
            <a
              href="#architecture"
              className="px-4 py-2 rounded-md border border-line hover:border-accent transition"
            >
              Architecture
            </a>
          </div>
        </div>
      </div>

      {/* KPI strip */}
      <section className="mx-auto max-w-6xl px-6 py-12">
        <h2 className="text-sm font-mono text-mute uppercase tracking-wider mb-4">
          Latest run · 14-day synthetic fleet
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <Kpi label="Raw events" value={fmt.format(k.raw_events)} />
          <Kpi label="Correlated incidents" value={fmt.format(k.incidents)} />
          <Kpi
            label="Alert compression"
            value={`${k.compression_x.toFixed(1)}×`}
            accent
          />
          <Kpi label="Critical incidents" value={fmt.format(k.critical_incidents)} />
          <Kpi label="Raw-event noise" value={fmtPct(k.noise_rate)} />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <Kpi label="Avg fleet draw" value={`${fmt.format(k.fleet_kw_avg)} kW`} />
          <Kpi label="Total energy" value={`${fmt.format(k.total_kwh)} kWh`} />
          <Kpi
            label="Idle waste"
            value={`${fmt.format(k.idle_kwh)} kWh`}
            sub={`$${fmt.format(k.idle_waste_dollars)} opportunity cost`}
          />
          <Kpi
            label="Performance / Watt"
            value={`${k.perf_per_watt_tflops.toFixed(4)} TFLOPS/W`}
          />
        </div>
      </section>

      {/* PdM */}
      <section id="pdm" className="mx-auto max-w-6xl px-6 py-12 border-t border-line">
        <h2 className="text-2xl font-semibold mb-2">Predictive Maintenance</h2>
        <p className="text-mute mb-6">
          Industry-grade pipeline: <span className="text-ink">hybrid time-series features</span>
          {" · "}calibrated HistGradientBoosting{" · "}
          Cox proportional-hazards survival{" · "}
          IsolationForest anomaly detection{" · "}
          rolling-origin TimeSeriesSplit{" · "}SHAP explanations{" · "}
          MLflow tracking{" · "}versioned registry{" · "}PSI drift detection.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <PdmCard
            label="ROC AUC"
            value={m.roc_auc_mean.toFixed(3)}
            delta={`± ${m.roc_auc_std.toFixed(3)}`}
          />
          <PdmCard label="PR AUC" value={m.pr_auc_mean.toFixed(3)} />
          <PdmCard
            label="Brier"
            value={m.brier_mean.toFixed(4)}
            hint="lower = better calibrated"
          />
          <PdmCard label="Precision @10" value={m.precision_at_10_mean.toFixed(2)} />
          <PdmCard
            label="Lift @10"
            value={`${m.lift_at_10_mean.toFixed(2)}×`}
            accent
          />
          <PdmCard
            label="Cox C-index"
            value={m.cox_c_index?.toFixed(3) ?? "—"}
          />
        </div>
        <p className="text-xs text-mute mt-4">
          Metrics are mean ± std across <strong>{m.n_folds} rolling-origin
          TimeSeriesSplit folds</strong>, trained on{" "}
          <strong>{fmt.format(m.n_training_rows)} rows × {m.n_features} features</strong>.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mt-8">
          <div className="panel p-6">
            <h3 className="font-semibold mb-4">Top 10 at-risk nodes (next 24h)</h3>
            <div className="space-y-2">
              {snap.top_at_risk.map((n) => (
                <div
                  key={n.node_id}
                  className="flex items-center justify-between font-mono text-sm border-b border-line/50 pb-2 last:border-0"
                >
                  <div className="flex gap-3">
                    <span className="text-ink">{n.node_id}</span>
                    <span className="text-mute">{n.rack_id}</span>
                    <span className="text-mute">{n.zone_id}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-mute">anom {n.anomaly_score.toFixed(2)}</span>
                    <span
                      className={`px-2 py-0.5 rounded ${
                        n.failure_risk_24h > 0.7
                          ? "bg-crit/20 text-crit"
                          : n.failure_risk_24h > 0.4
                          ? "bg-warn/20 text-warn"
                          : "bg-line text-mute"
                      }`}
                    >
                      {(n.failure_risk_24h * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel p-6">
            <h3 className="font-semibold mb-4">Top features (mean |SHAP|)</h3>
            <div className="space-y-3">
              {snap.top_features.slice(0, 8).map((f, i) => {
                const max = snap.top_features[0]?.mean_abs_shap ?? 1;
                const w = (f.mean_abs_shap / max) * 100;
                return (
                  <div key={f.feature}>
                    <div className="flex justify-between text-sm">
                      <span className="font-mono text-ink">{f.feature}</span>
                      <span className="text-mute font-mono">
                        {f.mean_abs_shap.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-1.5 mt-1 bg-line rounded-full overflow-hidden">
                      <div
                        className="h-full bg-accent"
                        style={{ width: `${w}%`, opacity: 1 - i * 0.07 }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-mute mt-4">
              <strong>ev_nvlink_fault_warn</strong> at the top confirms the simulator&apos;s
              precursor signal: nodes accumulating NVLink warns are the most likely to
              produce a critical failure in the next 24 hours.
            </p>
          </div>
        </div>
      </section>

      {/* Dashboard screenshot */}
      <section className="mx-auto max-w-6xl px-6 py-12 border-t border-line">
        <h2 className="text-2xl font-semibold mb-2">Live operator dashboard</h2>
        <p className="text-mute mb-6">
          Streamlit dashboard with eight panels — KPI strip, event mix, top incidents,
          energy & efficiency, capacity forecast, predictive maintenance, root-cause
          clusters, and utilization timeseries.
        </p>
        <div className="panel overflow-hidden">
          <Image
            src="/screenshots/dashboard-v3-final.png"
            alt="GridSentinel Streamlit dashboard — full view"
            width={2400}
            height={6000}
            className="w-full h-auto"
          />
        </div>
      </section>

      {/* Architecture */}
      <section id="architecture" className="mx-auto max-w-6xl px-6 py-12 border-t border-line">
        <h2 className="text-2xl font-semibold mb-2">Architecture</h2>
        <p className="text-mute mb-6">
          Closed loop: synthetic telemetry (or real DCGM/Prometheus/OTel) → enrichment →
          dedupe → correlate → cluster → forecast → energy → predictive model → registry
          → dashboard / API / weekly report.
        </p>
        <div className="panel p-6 overflow-x-auto">
          <pre className="text-xs font-mono text-mute leading-relaxed">{`simulator     ingest                                  forecast
   │             │                                       │
   │             ▼                                       ▼
   │      enrich_topology   correlate   root-cause    capacity
   │             │            dedupe     cluster      headroom
   │             ▼              ▼          ▼             │
   │       events.parquet  incidents.   HDBSCAN          │
   │             │           parquet   silhouette        │
   │             │              │       +0.75            │
   ▼             ▼              ▼          ▼             ▼
metrics.parquet ─┴──────────► PdM training pipeline ◄─────┘
                              ├─ hybrid TS features
                              ├─ HistGradientBoosting + isotonic calibration
                              ├─ Cox PH survival   (C-index 0.567)
                              ├─ IsolationForest anomaly
                              ├─ TimeSeriesSplit  (5 folds)
                              ├─ SHAP global + per-node
                              ├─ TCN deep baseline (benchmarked, loses)
                              └─ MLflow + versioned registry
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
       Streamlit dashboard       FastAPI service             Weekly LLM report
                                 (Docker, /at-risk,           (Claude API +
                                  /predict, /explain)         deterministic fallback)`}</pre>
        </div>
      </section>

      {/* Capabilities grid */}
      <section className="mx-auto max-w-6xl px-6 py-12 border-t border-line">
        <h2 className="text-2xl font-semibold mb-6">What's in the box</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <Capability
            title="Simulator"
            items={[
              "4,000-GPU fleet, 500 nodes × 8 GPUs",
              "8 correlated failure modes",
              "Stateful precursor patterns",
              "Continuous DCGM-style telemetry",
            ]}
          />
          <Capability
            title="Correlation + clustering"
            items={[
              "Topology-aware dedupe",
              "12× alert compression",
              "HDBSCAN root-cause families",
              "Silhouette 0.75",
            ]}
          />
          <Capability
            title="Predictive maintenance"
            items={[
              "Hybrid TS + event features (72 dim)",
              "Calibrated HistGradientBoosting",
              "Cox PH survival, IsolationForest",
              "TimeSeriesSplit + SHAP + drift PSI",
            ]}
          />
          <Capability
            title="Real-data ingest"
            items={[
              "DCGM CSV + dmon",
              "Prometheus query_range",
              "OpenTelemetry OTLP/JSON",
              "Live HTTP client for Mimir",
            ]}
          />
          <Capability
            title="Serving + APIs"
            items={[
              "FastAPI + Pydantic + OpenAPI",
              "Dockerfile, deployable image",
              "/predict, /explain, /at-risk",
              "Model registry-backed",
            ]}
          />
          <Capability
            title="Reporting + BI"
            items={[
              "Weekly LLM-narrative report",
              "Power BI star-schema + DAX",
              "Tableau CSV + Calculated Fields",
              "MLflow experiment tracking",
            ]}
          />
        </div>
      </section>

      {/* Bottom links */}
      <footer className="border-t border-line mt-12">
        <div className="mx-auto max-w-6xl px-6 py-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <div className="font-mono text-xs text-mute">
              data_hash {snap.data_hash} · {fmt.format(m.n_training_rows)} train rows
              · {m.n_features} features
            </div>
            <div className="text-mute text-sm mt-2">
              Built with Python · scikit-learn · PyTorch · lifelines · SHAP · MLflow ·
              FastAPI · Streamlit · Next.js.
            </div>
          </div>
          <div className="flex gap-4 text-sm">
            <a
              href="https://github.com/syedshabeebnibras/GridSentinel"
              className="text-accent hover:underline"
            >
              GitHub
            </a>
            <a
              href="https://github.com/syedshabeebnibras/GridSentinel/blob/main/README.md"
              className="text-accent hover:underline"
            >
              README
            </a>
            <a
              href="https://github.com/syedshabeebnibras/GridSentinel/tree/main/docs"
              className="text-accent hover:underline"
            >
              Architecture docs
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}

function Kpi({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className={`panel p-4 ${accent ? "ring-1 ring-accent/40" : ""}`}>
      <div className="text-xs text-mute uppercase tracking-wider">{label}</div>
      <div
        className={`kpi-num text-2xl md:text-3xl mt-1 ${
          accent ? "text-accent" : "text-ink"
        }`}
      >
        {value}
      </div>
      {sub && <div className="text-xs text-mute mt-1">{sub}</div>}
    </div>
  );
}

function PdmCard({
  label,
  value,
  delta,
  hint,
  accent,
}: {
  label: string;
  value: string;
  delta?: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className={`panel p-4 ${accent ? "ring-1 ring-accent/40" : ""}`}>
      <div className="text-xs text-mute">{label}</div>
      <div
        className={`kpi-num text-xl md:text-2xl mt-1 ${
          accent ? "text-accent" : "text-ink"
        }`}
      >
        {value}
      </div>
      {delta && <div className="text-xs text-mute mt-0.5 font-mono">{delta}</div>}
      {hint && <div className="text-[10px] text-mute mt-1 leading-tight">{hint}</div>}
    </div>
  );
}

function Capability({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="panel p-5">
      <div className="text-sm font-semibold text-ink mb-3">{title}</div>
      <ul className="space-y-1.5 text-sm text-mute">
        {items.map((it) => (
          <li key={it} className="flex items-start gap-2">
            <span className="text-accent mt-0.5">▸</span>
            <span>{it}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
