"""SLA KPIs — availability, MTTR, MTBF, MTTD, incident rate."""
from __future__ import annotations

import pandas as pd


def availability(node_hours_total: float, unplanned_downtime_hours: float) -> float:
    """A = (node-hours − downtime) / node-hours.   Range [0, 1]."""
    if node_hours_total <= 0:
        return 0.0
    return max(0.0, (node_hours_total - unplanned_downtime_hours) / node_hours_total)


def mtbf(operating_hours: float, failure_count: int) -> float:
    """Mean Time Between Failures (hours)."""
    if failure_count <= 0:
        return float("inf")
    return operating_hours / failure_count


def mttr(tickets: pd.DataFrame) -> float:
    """Expects columns: opened_at, resolved_at (datetime64). Returns hours."""
    if tickets.empty:
        return 0.0
    delta = (tickets["resolved_at"] - tickets["opened_at"]).dt.total_seconds() / 3600
    return float(delta.mean())


def mttd(tickets: pd.DataFrame) -> float:
    """Mean Time To Detect — event_at → opened_at. Hours."""
    if tickets.empty:
        return 0.0
    delta = (tickets["opened_at"] - tickets["event_at"]).dt.total_seconds() / 3600
    return float(delta.mean())


def incident_rate(incident_count: int, node_hours_total: float) -> float:
    """Incidents per 1,000 node-hours."""
    if node_hours_total <= 0:
        return 0.0
    return 1000.0 * incident_count / node_hours_total


def first_time_resolution(tickets: pd.DataFrame) -> float:
    """(resolved − reopened) / resolved. Expects bool column `reopened`."""
    resolved = tickets["resolved_at"].notna().sum()
    if resolved == 0:
        return 0.0
    reopened = int(tickets["reopened"].sum()) if "reopened" in tickets.columns else 0
    return (resolved - reopened) / resolved


def alert_compression_ratio(raw_alerts: int, correlated_incidents: int) -> float:
    """Goal: ≥ 10×. Higher is better noise reduction."""
    if correlated_incidents == 0:
        return float("inf")
    return raw_alerts / correlated_incidents


def noise_rate(benign_alerts: int, total_alerts: int) -> float:
    """Fraction of raw alerts that turned out to be benign."""
    if total_alerts == 0:
        return 0.0
    return benign_alerts / total_alerts
