"""Convert raw events into ticket-shaped rows for SLA / MTTR analysis.

A ticket = one correlated incident with opened_at, resolved_at, severity,
component, and assigned_team. Built from correlated events (see
gridsentinel.correlation).
"""
from __future__ import annotations

import pandas as pd

# TODO: implement once correlation pipeline is in place.
def to_tickets(correlated_incidents: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
