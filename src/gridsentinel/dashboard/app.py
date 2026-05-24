"""Streamlit dashboard. Run with: streamlit run src/gridsentinel/dashboard/app.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from gridsentinel.correlation.correlate import correlate
from gridsentinel.correlation.dedupe import dedupe
from gridsentinel.forecast.capacity import headroom_pct, weeks_to_exhaustion
from gridsentinel.forecast.utilization import forecast_gpu_hours, hourly_gpu_hours
from gridsentinel.ingest.telemetry import enrich_with_topology
from gridsentinel.kpi.efficiency import fleet_energy_summary, power_timeseries
from gridsentinel.kpi.sla import alert_compression_ratio, noise_rate
from gridsentinel.predict.registry import load_latest as load_latest_model
from gridsentinel.rootcause.cluster import cluster_incidents, cluster_quality, top_recurring
from gridsentinel.rootcause.features import featurize

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"

st.set_page_config(page_title="GridSentinel", layout="wide")
st.title("GridSentinel — GPU Fleet Ops")

events_path = DATA_DIR / "events.parquet"
util_path = DATA_DIR / "utilization.parquet"
fleet_path = DATA_DIR / "fleet.parquet"

if not events_path.exists() or not fleet_path.exists():
    st.warning("No simulation data yet. Run: `python -m gridsentinel.simulator.emit`")
    st.stop()


@st.cache_data(show_spinner="Loading data, building incidents + features...")
def load_all():
    from gridsentinel.predict.ts_features import build_timeseries_dataset

    events = pd.read_parquet(events_path)
    util = pd.read_parquet(util_path)
    fleet = pd.read_parquet(fleet_path)
    enriched = enrich_with_topology(events, fleet)
    deduped = dedupe(enriched, window_ticks=5)
    incidents = correlate(deduped, time_window_ticks=15)
    feats = featurize(incidents)
    clustered = cluster_incidents(feats, incidents, min_cluster_size=8)
    sil = cluster_quality(feats, clustered["cluster_id"])

    # Latest scoring window for the PdM leaderboard. Prefer the on-disk
    # artifact persisted by `python -m gridsentinel.predict.score` — that's
    # ≤ 500 rows and loads in milliseconds. Only rebuild from raw 16M
    # samples if the artifact is missing (slow path).
    latest_parquet = (
        Path(__file__).resolve().parents[3] / "data" / "predict" / "latest_window.parquet"
    )
    metrics_p = DATA_DIR / "metrics.parquet"
    if latest_parquet.exists():
        latest_window = pd.read_parquet(latest_parquet)
    elif metrics_p.exists():
        metrics_df = pd.read_parquet(metrics_p)
        ts_ds = build_timeseries_dataset(metrics_df, enriched, fleet)
        if not ts_ds.empty:
            latest_window = ts_ds[ts_ds["window_end_tick"] == ts_ds["window_end_tick"].max()].copy()
        else:
            latest_window = pd.DataFrame()
    else:
        latest_window = pd.DataFrame()

    pdm_registry = load_latest_model()
    return (
        events, util, fleet, enriched, deduped, incidents, clustered, sil,
        pdm_registry, latest_window,
    )


(
    events, util, fleet, enriched, deduped, incidents, clustered, sil,
    pdm_registry, latest_window,
) = load_all()

# --- KPI strip --------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Raw events", f"{len(events):,}")
c2.metric("After dedupe", f"{len(deduped):,}")
c3.metric("Incidents", f"{len(incidents):,}")
c4.metric(
    "Compression",
    f"{alert_compression_ratio(len(events), len(incidents)):.1f}×",
)
c5.metric("Noise rate", f"{noise_rate(int(events['benign'].sum()), len(events)):.1%}")

# --- Event mix --------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Event kinds")
    st.bar_chart(events.groupby("kind").size())
with right:
    st.subheader("Severity (raw vs incidents)")
    sev = pd.DataFrame(
        {
            "raw": events["severity"].value_counts(),
            "incidents": incidents["severity_max"].value_counts(),
        }
    ).fillna(0)
    st.bar_chart(sev)

# --- Top incidents ----------------------------------------------------------
st.subheader("Top 10 incidents by member count")
st.dataframe(
    incidents.nlargest(10, "member_count")[
        [
            "incident_id",
            "root_kind",
            "scope",
            "member_count",
            "severity_max",
            "any_benign",
            "opened_tick",
            "duration_ticks",
        ]
    ],
    use_container_width=True,
)

# --- Utilization ------------------------------------------------------------
st.subheader("Mean GPU utilization over time")
st.line_chart(util.groupby("tick")["util"].mean())

# --- Energy & efficiency ---------------------------------------------------
st.subheader("Energy & efficiency (IREN view)")
energy = fleet_energy_summary(util, gpus_per_node=8)
e1, e2, e3, e4 = st.columns(4)
e1.metric("Avg fleet draw", f"{energy['avg_fleet_kw']:,.0f} kW")
e2.metric("Total energy", f"{energy['total_kwh']:,.0f} kWh")
e3.metric(
    "Idle waste",
    f"{energy['idle_kwh']:,.0f} kWh",
    delta=f"${energy['idle_waste_dollars']:,.0f}",
    delta_color="inverse",
)
e4.metric("Perf / Watt", f"{energy['perf_per_watt_tflops']:.4f} TFLOPS/W")

power_ts = power_timeseries(util, gpus_per_node=8)
st.line_chart(power_ts.set_index("tick")["fleet_kw"], height=180)
st.caption(
    "Each kWh of idle draw is paid for whether or not it produces work — under "
    "a renewable PPA, it represents both wasted spend and wasted clean energy."
)

# --- Capacity forecast -----------------------------------------------------
st.subheader("Capacity forecast (next 4 weeks)")
hourly = hourly_gpu_hours(util)
forecast = forecast_gpu_hours(hourly, horizon_hours=28 * 24)
# Max GPU-hours per hour = total GPUs (4000 nominal) × 1 hour
max_per_hour = float(fleet["node_id"].nunique() * 8)
weeks = weeks_to_exhaustion(forecast, max_per_hour, threshold=0.70)
current_avg = float(hourly.mean()) if len(hourly) else 0.0
hp = headroom_pct(current_avg, max_per_hour)

f1, f2, f3 = st.columns(3)
f1.metric("Fleet capacity", f"{int(max_per_hour):,} GPU-h/h")
f2.metric("Current headroom", f"{hp:.1%}")
f3.metric(
    "Weeks to 70% load",
    "∞" if weeks is None else f"{weeks:.1f}",
)
forecast_chart = pd.concat(
    [
        hourly.rename("observed").reset_index().assign(series="observed"),
        forecast.rename(columns={"yhat": "observed"})[["tick", "observed"]].assign(
            series="forecast"
        ),
    ]
)
st.line_chart(
    forecast_chart.pivot(index="tick", columns="series", values="observed"),
    height=200,
)

st.subheader("Top recurring root-cause families")
n_clusters = int((clustered["cluster_id"] != -1).sum()) and clustered["cluster_id"].nunique()
n_noise = int((clustered["cluster_id"] == -1).sum())
sil_text = f"silhouette {sil:+.2f}" if sil is not None else "silhouette n/a"
st.caption(
    f"HDBSCAN found {n_clusters} clusters across {len(clustered):,} incidents "
    f"({n_noise:,} flagged as noise/unclustered) · {sil_text}."
)
st.dataframe(top_recurring(clustered, n=10), use_container_width=True)

# --- Predictive maintenance (PdM) — v2 industry-grade ---------------------
st.subheader("Predictive maintenance — 24h node failure risk")
if pdm_registry is None:
    st.info(
        "No registered PdM model. Run "
        "`python -m gridsentinel.predict.score` to train + register, then "
        "refresh this dashboard."
    )
else:
    meta = pdm_registry["metadata"]
    cal_model = pdm_registry["calibrated"]
    survival_model = pdm_registry["survival"]
    anom_model = pdm_registry["anomaly"]

    st.caption(
        f"Registry: **{pdm_registry['version']}** · {meta['n_training_rows']:,} rows × "
        f"{meta['n_features']} features · data_hash={meta['data_hash']} · "
        f"trained {meta['created_at']}"
    )

    m = cal_model.metrics
    pdm_cols = st.columns(6)
    pdm_cols[0].metric(
        "ROC AUC",
        f"{m.get('roc_auc_mean', 0):.3f}",
        delta=f"±{m.get('roc_auc_std', 0):.3f}",
        delta_color="off",
    )
    pdm_cols[1].metric("PR AUC", f"{m.get('pr_auc_mean', 0):.3f}")
    pdm_cols[2].metric("Brier", f"{m.get('brier_mean', 0):.4f}")
    pdm_cols[3].metric("Precision @10", f"{m.get('precision_at_10_mean', 0):.2f}")
    pdm_cols[4].metric("Lift @10", f"{m.get('lift_at_10_mean', 0):.2f}×")
    if survival_model is not None and survival_model.estimator is not None:
        pdm_cols[5].metric("Cox C-index", f"{survival_model.metrics.get('c_index', 0):.3f}")
    else:
        pdm_cols[5].metric("Cox C-index", "n/a")

    st.caption(
        f"Metrics are mean ± std across **{m.get('n_folds', 0)} rolling-origin "
        "TimeSeriesSplit folds**. Calibrated with isotonic regression on the "
        "training tail. Brier score < 0.20 ≈ well-calibrated probabilities."
    )

    # Score the latest window (pre-computed in load_all + cached).
    latest = latest_window
    if not latest.empty:
        latest_X = latest[cal_model.feature_names]
        proba = cal_model.predict_proba(latest_X)
        leaderboard = latest[["node_id", "rack_id", "zone_id"]].copy()
        leaderboard["failure_risk_24h"] = proba
        # add anomaly score (defense-in-depth)
        leaderboard["anomaly_score"] = anom_model.score(latest_X)
        leaderboard = leaderboard.sort_values("failure_risk_24h", ascending=False).head(10).reset_index(drop=True)

        l, r = st.columns([3, 2])
        with l:
            st.markdown("**Top 10 at-risk nodes (next 24h)** — supervised + anomaly")
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        with r:
            st.markdown("**Top features (mean |SHAP|)**")
            if cal_model.shap_summary is not None:
                st.dataframe(
                    cal_model.shap_summary.head(8),
                    use_container_width=True, hide_index=True,
                )

        # SHAP explanation for the top at-risk node
        if cal_model.shap_summary is not None:
            from gridsentinel.predict.calibrated import shap_explanation_for_node
            top_node_row = latest.loc[latest["node_id"] == leaderboard.iloc[0]["node_id"]].head(1)
            if not top_node_row.empty:
                shap_row = shap_explanation_for_node(cal_model, top_node_row, top_n=6)
                if not shap_row.empty:
                    st.markdown(f"**Why is {leaderboard.iloc[0]['node_id']} the riskiest? (per-prediction SHAP)**")
                    st.dataframe(shap_row, use_container_width=True, hide_index=True)

        # Drift indicator panel — compare latest window distribution vs
        # the training-time baseline that was registered alongside the model.
        from gridsentinel.predict.drift import psi_report
        try:
            drift_df = psi_report(pdm_registry["drift_baseline"], latest, pdm_registry["drift_baseline"])
            high_drift = drift_df[drift_df["severity"] != "stable"]
            if not high_drift.empty:
                st.markdown(f"**Feature drift** — {len(high_drift)} feature(s) showing PSI shift")
                st.dataframe(high_drift.head(10), use_container_width=True, hide_index=True)
            else:
                st.success(f"Feature drift: no shifts detected (all {len(drift_df)} features stable, PSI < 0.10)")
        except Exception:
            pass

    st.caption(
        "Pipeline: HistGradientBoosting + isotonic calibration, Cox proportional-hazards "
        "survival model (lifelines), IsolationForest anomaly detector, rolling-origin "
        "TSCV, SHAP explanations, MLflow tracking, versioned model registry, PSI drift."
    )
