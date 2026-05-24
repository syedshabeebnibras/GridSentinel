"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "https://gridsentinel-api-production.up.railway.app";

type Version = {
  model_version: string;
  n_training_rows: number;
  n_features: number;
  metrics: Record<string, number>;
  data_hash: string;
  trained_at: string;
};

type AtRiskNode = {
  node_id: string;
  rack_id: string | null;
  zone_id: string | null;
  failure_risk_24h: number;
  anomaly_score: number;
};

type State =
  | { status: "loading" }
  | { status: "ok"; version: Version; nodes: AtRiskNode[]; fetchedAt: number; latencyMs: number }
  | { status: "offline"; error: string };

const POLL_MS = 30_000;
const FETCH_TIMEOUT_MS = 5_000;

async function fetchWithTimeout(url: string, ms: number): Promise<Response> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { signal: ctrl.signal, cache: "no-store" });
  } finally {
    clearTimeout(t);
  }
}

async function pull(): Promise<State> {
  const started = performance.now();
  try {
    const [vRes, rRes] = await Promise.all([
      fetchWithTimeout(`${API}/version`, FETCH_TIMEOUT_MS),
      fetchWithTimeout(`${API}/at-risk?n=10`, FETCH_TIMEOUT_MS),
    ]);
    if (!vRes.ok || !rRes.ok) throw new Error(`HTTP ${vRes.status}/${rRes.status}`);
    const version = (await vRes.json()) as Version;
    const atRisk = (await rRes.json()) as { nodes: AtRiskNode[] };
    return {
      status: "ok",
      version,
      nodes: atRisk.nodes,
      fetchedAt: Date.now(),
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (e: unknown) {
    return { status: "offline", error: e instanceof Error ? e.message : "unknown" };
  }
}

export function LiveStatus() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const next = await pull();
      if (!cancelled) {
        setState(next);
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
      <div className="flex items-center gap-2 text-mute text-xs font-mono">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-mute animate-pulse" />
        <span>connecting to API…</span>
      </div>
    );
  }

  if (state.status === "offline") {
    return (
      <div className="flex items-center gap-2 text-warn text-xs font-mono">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-warn" />
        <span>API offline — showing baked-in snapshot</span>
      </div>
    );
  }

  const ageS = Math.floor((Date.now() - state.fetchedAt) / 1000);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs font-mono">
      <span className="flex items-center gap-2 text-accent">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shadow-[0_0_8px_rgba(94,234,212,0.7)] animate-pulse" />
        <span>LIVE · {state.latencyMs} ms</span>
      </span>
      <span className="text-mute">model {state.version.model_version}</span>
      <span className="text-mute">{state.version.n_features} features</span>
      <span className="text-mute">refreshed {ageS}s ago</span>
    </div>
  );
}

export function LiveAtRisk({ fallback }: { fallback: AtRiskNode[] }) {
  const [nodes, setNodes] = useState<AtRiskNode[]>(fallback);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const r = await pull();
      if (!cancelled && r.status === "ok") {
        setNodes(r.nodes);
        setLive(true);
        timer = setTimeout(tick, POLL_MS);
      } else if (!cancelled) {
        setLive(false);
        timer = setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Top 10 at-risk nodes (next 24h)</h3>
        {live ? (
          <span className="text-[10px] font-mono uppercase tracking-wider text-accent flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            live
          </span>
        ) : (
          <span className="text-[10px] font-mono uppercase tracking-wider text-mute">
            cached snapshot
          </span>
        )}
      </div>
      <div className="space-y-2">
        {nodes.map((n) => (
          <div
            key={n.node_id}
            className="flex items-center justify-between font-mono text-sm border-b border-line/50 pb-2 last:border-0"
          >
            <div className="flex gap-3">
              <span className="text-ink">{n.node_id}</span>
              <span className="text-mute">{n.rack_id ?? ""}</span>
              <span className="text-mute">{n.zone_id ?? ""}</span>
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
  );
}
