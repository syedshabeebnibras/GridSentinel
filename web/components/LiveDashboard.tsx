"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "https://gridsentinel-api-production.up.railway.app";

type Summary = {
  event_kinds: { kind: string; count: number }[];
  severity: {
    raw: { info: number; warn: number; critical: number };
    incidents: { info: number; warn: number; critical: number };
  };
  top_incidents: {
    incident_id: string;
    root_kind: string;
    scope: string;
    member_count: number;
    severity_max: string;
    any_benign: boolean;
  }[];
  activity_series: { tick: number; activity: number }[];
  totals: { raw_events: number; incidents: number; deduped: number };
};

type State =
  | { status: "loading" }
  | { status: "ok"; data: Summary; fetchedAt: number; latencyMs: number }
  | { status: "offline"; error: string };

const POLL_MS = 60_000;
const fmt = new Intl.NumberFormat("en-US");

async function pullSummary(): Promise<State> {
  const t0 = performance.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);
    const r = await fetch(`${API}/summary`, { signal: ctrl.signal, cache: "no-store" });
    clearTimeout(timer);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = (await r.json()) as Summary;
    return {
      status: "ok",
      data,
      fetchedAt: Date.now(),
      latencyMs: Math.round(performance.now() - t0),
    };
  } catch (e: unknown) {
    return { status: "offline", error: e instanceof Error ? e.message : "unknown" };
  }
}

export function LiveDashboard() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const s = await pullSummary();
      if (!cancelled) {
        setState(s);
        timer = setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  if (state.status === "loading") {
    return (
      <div className="panel p-12 text-center text-mute font-mono text-sm">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-mute animate-pulse mr-2" />
        loading live dashboard from API…
      </div>
    );
  }

  if (state.status === "offline") {
    return (
      <div className="panel p-12 text-center text-warn font-mono text-sm">
        API offline — {state.error}. Run{" "}
        <code className="text-ink">streamlit run src/gridsentinel/dashboard/app.py</code>{" "}
        locally to see the full eight-panel Streamlit equivalent.
      </div>
    );
  }

  const { data } = state;
  const compression = data.totals.raw_events / Math.max(data.totals.incidents, 1);

  return (
    <div className="panel p-6 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <div>
          <h3 className="text-lg font-semibold">GridSentinel — GPU Fleet Ops</h3>
          <p className="text-xs text-mute mt-1">
            Same panels as the Streamlit app, hydrated from <code>GET /summary</code>.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-accent">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          LIVE · {state.latencyMs} ms · refreshed{" "}
          {Math.floor((Date.now() - state.fetchedAt) / 1000)}s ago
        </div>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
        <KPI label="Raw events" value={fmt.format(data.totals.raw_events)} />
        <KPI label="After dedupe" value={fmt.format(data.totals.deduped)} />
        <KPI label="Incidents" value={fmt.format(data.totals.incidents)} />
        <KPI label="Compression" value={`${compression.toFixed(1)}×`} accent />
        <KPI
          label="Critical"
          value={fmt.format(data.severity.incidents.critical)}
          danger={data.severity.incidents.critical > 0}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <Panel title="Event kinds">
          <EventKindsBars data={data.event_kinds} />
        </Panel>
        <Panel title="Severity (raw vs incidents)">
          <SeverityStacked
            raw={data.severity.raw}
            incidents={data.severity.incidents}
          />
        </Panel>
      </div>

      <div className="mb-8">
        <Panel title="Top 10 incidents by member count">
          <TopIncidentsTable rows={data.top_incidents} />
        </Panel>
      </div>

      <div>
        <Panel title="Fleet activity over time (severity-weighted)">
          <ActivitySpark series={data.activity_series} />
        </Panel>
      </div>
    </div>
  );
}

function KPI({
  label,
  value,
  accent,
  danger,
}: {
  label: string;
  value: string;
  accent?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="bg-bg/40 border border-line rounded-lg p-3">
      <div className="text-[10px] uppercase tracking-wider text-mute">{label}</div>
      <div
        className={`kpi-num text-xl mt-1 ${
          accent ? "text-accent" : danger ? "text-crit" : "text-ink"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-bg/40 border border-line rounded-lg p-5">
      <div className="text-xs font-semibold uppercase tracking-wider text-mute mb-4">
        {title}
      </div>
      {children}
    </div>
  );
}

function EventKindsBars({ data }: { data: Summary["event_kinds"] }) {
  const max = Math.max(...data.map((d) => d.count), 1);
  return (
    <div className="space-y-2">
      {data.map((d) => (
        <div key={d.kind}>
          <div className="flex justify-between text-xs font-mono">
            <span className="text-ink">{d.kind}</span>
            <span className="text-mute">{fmt.format(d.count)}</span>
          </div>
          <div className="h-2 mt-1 bg-line rounded-full overflow-hidden">
            <div
              className="h-full bg-accent"
              style={{ width: `${(d.count / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function SeverityStacked({
  raw,
  incidents,
}: {
  raw: { info: number; warn: number; critical: number };
  incidents: { info: number; warn: number; critical: number };
}) {
  const rows: { label: string; mix: typeof raw }[] = [
    { label: "Raw events", mix: raw },
    { label: "Incidents", mix: incidents },
  ];
  return (
    <div className="space-y-4">
      {rows.map(({ label, mix }) => {
        const total = mix.info + mix.warn + mix.critical || 1;
        return (
          <div key={label}>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-ink font-mono">{label}</span>
              <span className="text-mute font-mono">{fmt.format(total)}</span>
            </div>
            <div className="h-3 flex rounded-full overflow-hidden bg-line">
              <Segment width={(mix.info / total) * 100} className="bg-mute" />
              <Segment width={(mix.warn / total) * 100} className="bg-warn" />
              <Segment width={(mix.critical / total) * 100} className="bg-crit" />
            </div>
            <div className="flex gap-3 text-[10px] font-mono mt-1 text-mute">
              <span>info {fmt.format(mix.info)}</span>
              <span className="text-warn">warn {fmt.format(mix.warn)}</span>
              <span className="text-crit">crit {fmt.format(mix.critical)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Segment({ width, className }: { width: number; className: string }) {
  if (width < 0.05) return null;
  return <div className={className} style={{ width: `${width}%` }} />;
}

function TopIncidentsTable({ rows }: { rows: Summary["top_incidents"] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-mute uppercase tracking-wider text-[10px]">
            <th className="text-left py-2 pr-3">Incident</th>
            <th className="text-left pr-3">Kind</th>
            <th className="text-left pr-3">Scope</th>
            <th className="text-right pr-3">Members</th>
            <th className="text-left">Severity</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.incident_id} className="border-t border-line/40">
              <td className="py-1.5 pr-3 text-ink truncate max-w-[280px]">
                {r.incident_id}
              </td>
              <td className="pr-3 text-mute">{r.root_kind}</td>
              <td className="pr-3 text-mute">{r.scope}</td>
              <td className="pr-3 text-right text-ink">{r.member_count}</td>
              <td>
                <span
                  className={`px-1.5 py-0.5 rounded ${
                    r.severity_max === "critical"
                      ? "bg-crit/20 text-crit"
                      : r.severity_max === "warn"
                      ? "bg-warn/20 text-warn"
                      : "bg-line text-mute"
                  }`}
                >
                  {r.severity_max}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ActivitySpark({ series }: { series: Summary["activity_series"] }) {
  if (series.length < 2) return <div className="text-mute text-sm">no data</div>;
  const W = 800;
  const H = 140;
  const PAD = 10;
  const values = series.map((p) => p.activity);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = (W - PAD * 2) / (series.length - 1);
  const points = series.map((p, i) => {
    const x = PAD + i * stepX;
    const y = H - PAD - ((p.activity - min) / span) * (H - PAD * 2);
    return [x, y] as const;
  });
  const linePath = "M" + points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L");
  const areaPath = `${linePath} L${(W - PAD).toFixed(1)},${H - PAD} L${PAD},${H - PAD} Z`;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-32" preserveAspectRatio="none">
        <defs>
          <linearGradient id="actgrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#5eead4" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#5eead4" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#actgrad)" />
        <path d={linePath} fill="none" stroke="#5eead4" strokeWidth="1.5" />
      </svg>
      <div className="flex justify-between text-[10px] font-mono text-mute mt-1">
        <span>tick {series[0].tick}</span>
        <span>min {fmt.format(min)} · max {fmt.format(max)}</span>
        <span>tick {series[series.length - 1].tick}</span>
      </div>
    </div>
  );
}
