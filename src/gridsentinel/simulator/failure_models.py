"""Failure models — the domain-knowledge core of the simulator.

Each failure mode is a function of:
  - a per-tick base hazard rate (lambda)
  - modulators (load, temperature, age, time-of-day)
  - a CORRELATION KEY (gpu / node / rack / zone / feed / spine)

When a parent component (e.g. a zone's CRAC) fails, all children sharing that
correlation key experience the downstream failure simultaneously. That is what
makes the simulator realistic — and what makes the correlation engine actually
have something to compress.

Each model returns FailureEvents which the emitter writes to the telemetry +
ticket streams. The correlation engine downstream tries to recover the parent.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from gridsentinel.simulator.topology import Fleet, Node


class FailureKind(str, Enum):
    THERMAL_THROTTLE = "thermal_throttle"
    ECC_UNCORRECTABLE = "ecc_uncorrectable"
    NVLINK_FAULT = "nvlink_fault"
    PCIE_ERROR = "pcie_error"
    PSU_TRIP = "psu_trip"
    NETWORK_FLAP = "network_flap"
    COOLING_FAILURE = "cooling_failure"
    POWER_EVENT = "power_event"


@dataclass
class FailureEvent:
    tick: int
    kind: FailureKind
    scope: str        # "gpu" | "node" | "rack" | "zone" | "feed" | "spine"
    target: str       # the id at that scope
    severity: str     # "info" | "warn" | "critical"
    benign: bool      # if True, this is noise — correlation should drop it
    parent_event_id: str | None = None  # links child failures to their parent


# ---------------------------------------------------------------------------
# Cooling failure — example of a parent (zone-level) event with cascading
# child throttle events on every node in the zone. Use this as the template
# for the modes you write below.
# ---------------------------------------------------------------------------
def cooling_failure(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    hazard_per_zone_per_hour: float = 1.0 / (30 * 24),  # ~1 event per zone per month
    ticks_per_hour: int = 60,
) -> list[FailureEvent]:
    """Zone-level CRAC alarm; cascades to thermal throttle on every node in zone."""
    out: list[FailureEvent] = []
    per_tick_hazard = hazard_per_zone_per_hour / ticks_per_hour
    zones = {n.zone_id for n in fleet.nodes}
    for zone in zones:
        if rng.random() < per_tick_hazard:
            parent_id = f"cooling-{zone}-{tick}"
            out.append(
                FailureEvent(
                    tick=tick,
                    kind=FailureKind.COOLING_FAILURE,
                    scope="zone",
                    target=zone,
                    severity="critical",
                    benign=False,
                    parent_event_id=parent_id,  # parent self-references to join cascade
                )
            )
            # cascade: every node in the zone throttles
            for node in fleet.by_zone(zone):
                out.append(
                    FailureEvent(
                        tick=tick,
                        kind=FailureKind.THERMAL_THROTTLE,
                        scope="node",
                        target=node.node_id,
                        severity="warn",
                        benign=False,
                        parent_event_id=parent_id,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# TODO(you) — write `thermal_throttle` for the *non-cooling-induced* case.
#
# This is the standalone version: a single GPU throttles because its own
# inlet temp × utilization crosses a threshold. NOT a cascade — this is the
# spontaneous, uncorrelated baseline that lives alongside the zone cascades.
#
# Trade-offs to think about:
#   (a) Stateless: `if rng.random() < lambda * util_factor: emit`
#         - simple, but every tick is independent → unrealistic burstiness
#   (b) Stateful: a per-GPU temperature random walk that crosses a threshold
#         - more realistic; gives the correlation engine a "warming up" signal
#         - requires a `gpu_state` dict you thread through ticks
#
# Recommended: option (b) for one mode (this one) to prove you understand
# stateful simulation; option (a) is fine for ECC / PCIe / NVLink elsewhere.
#
# Signature to implement:
#
# def thermal_throttle(
#     fleet: Fleet,
#     tick: int,
#     rng: random.Random,
#     gpu_state: dict[str, dict],   # mutable per-GPU state
#     load_by_gpu: dict[str, float],  # 0..1 utilization
# ) -> list[FailureEvent]:
#     ...
#
# Notes:
#   - ~70% of standalone events should have benign=True (transient spike,
#     auto-recovers within ~2 ticks). The correlation engine learns to drop
#     these by looking at duration.
#   - Severity ladder: temp 80-85C → "info" benign, 85-90C → "warn",
#     >90C sustained → "critical".
# ---------------------------------------------------------------------------
def thermal_throttle(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    gpu_state: dict[str, dict],
    load_by_gpu: dict[str, float],
) -> list[FailureEvent]:
    """Stateful per-GPU temperature walk; emit on state transition into throttle.

    Each GPU has a temperature that drifts toward a load-driven target with
    first-order lag + jitter. We emit a FailureEvent only when a GPU crosses
    from non-throttling → throttling (avoids flooding for sustained events).

    Benign-rate target is ~70% overall: severity tiers are info / warn / critical;
    most info-tier crossings are tagged benign (transient spike that auto-recovers).
    """
    AMBIENT = 22.0
    THRESHOLD = 85.0
    out: list[FailureEvent] = []

    for node in fleet.nodes:
        for gpu in node.gpus:
            gid = gpu.gpu_id
            load = load_by_gpu.get(gid, 0.0)
            st = gpu_state.setdefault(gid, {"temp": 60.0, "throttling": False})

            target = AMBIENT + 35.0 + 35.0 * load
            st["temp"] += 0.15 * (target - st["temp"]) + rng.gauss(0, 1.5)
            temp = st["temp"]

            was_throttling = st["throttling"]
            is_throttling = temp > THRESHOLD
            st["throttling"] = is_throttling

            if is_throttling and not was_throttling:
                if temp > 92.0:
                    sev, benign = "critical", False
                elif temp > 88.0:
                    sev = "warn"
                    benign = rng.random() < 0.30
                else:
                    sev = "info"
                    benign = rng.random() < 0.85

                out.append(
                    FailureEvent(
                        tick=tick,
                        kind=FailureKind.THERMAL_THROTTLE,
                        scope="gpu",
                        target=gid,
                        severity=sev,
                        benign=benign,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Stubs for the remaining modes. Fill out after thermal_throttle is working.
# ---------------------------------------------------------------------------
def ecc_uncorrectable(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    ecc_state: dict[str, int] | None = None,
    ticks_per_hour: int = 12,
) -> list[FailureEvent]:
    """GPU-scope ECC. Corrected errors accumulate per-GPU; once a GPU has
    seen `escalation_threshold` corrected errors, the next ECC event becomes
    uncorrectable (critical). This creates a real predictive signal — the
    classifier can learn that a rising ECC count precedes a critical event.

    `ecc_state` is a mutable per-GPU counter dict; pass the same dict across
    ticks. If None, escalation is disabled (random 5% uncorrectable).
    """
    out: list[FailureEvent] = []
    hazard = 1.0 / (600.0 * ticks_per_hour)
    escalation_threshold = 1
    for node in fleet.nodes:
        for gpu in node.gpus:
            if rng.random() < hazard:
                gid = gpu.gpu_id
                if ecc_state is not None:
                    prior = ecc_state.get(gid, 0)
                    uncorrectable = prior >= escalation_threshold and rng.random() < 0.95
                    if uncorrectable:
                        ecc_state[gid] = 0  # reset after critical
                    else:
                        ecc_state[gid] = prior + 1
                else:
                    uncorrectable = rng.random() < 0.05
                out.append(
                    FailureEvent(
                        tick=tick,
                        kind=FailureKind.ECC_UNCORRECTABLE,
                        scope="gpu",
                        target=gid,
                        severity="critical" if uncorrectable else "info",
                        benign=not uncorrectable,
                    )
                )
    return out


def nvlink_fault(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    nvlink_state: dict[str, int] | None = None,
    ticks_per_hour: int = 12,
) -> list[FailureEvent]:
    """Node-scope NVLink fault. Warns accumulate per-node; once 2+ warns have
    fired on the same node, the next event has elevated probability of being
    critical (real physics — degrading lanes precede outright failure).
    """
    out: list[FailureEvent] = []
    hazard = 0.5 / (24 * ticks_per_hour)
    for node in fleet.nodes:
        if rng.random() < hazard:
            warn_count = (nvlink_state or {}).get(node.node_id, 0)
            if nvlink_state is not None and warn_count >= 2:
                # escalation: 70% critical, 30% warn (and reset)
                severity = "critical" if rng.random() < 0.7 else "warn"
                if severity == "critical":
                    nvlink_state[node.node_id] = 0
            else:
                severity = rng.choices(["info", "warn", "critical"], weights=[0.55, 0.35, 0.10])[0]
                if nvlink_state is not None and severity == "warn":
                    nvlink_state[node.node_id] = warn_count + 1
            out.append(
                FailureEvent(
                    tick=tick,
                    kind=FailureKind.NVLINK_FAULT,
                    scope="node",
                    target=node.node_id,
                    severity=severity,
                    benign=severity == "info",
                )
            )
    return out


def pcie_error(
    fleet: Fleet, tick: int, rng: random.Random, ticks_per_hour: int = 12
) -> list[FailureEvent]:
    """Node-scope. AER counter increments — usually benign, occasionally critical."""
    out: list[FailureEvent] = []
    hazard = 0.3 / (24 * ticks_per_hour)
    for node in fleet.nodes:
        if rng.random() < hazard:
            severity = rng.choices(["info", "warn", "critical"], weights=[0.75, 0.2, 0.05])[0]
            out.append(
                FailureEvent(
                    tick=tick,
                    kind=FailureKind.PCIE_ERROR,
                    scope="node",
                    target=node.node_id,
                    severity=severity,
                    benign=severity == "info",
                )
            )
    return out


def psu_trip(
    fleet: Fleet,
    tick: int,
    rng: random.Random,
    rack_thermal_stress: dict[str, float] | None = None,
    ticks_per_hour: int = 12,
) -> list[FailureEvent]:
    """Rack-scope: all 5 nodes on the rack go dark together.

    Hazard scales with rolling rack thermal stress (real physics — sustained
    heat shortens PSU life). `rack_thermal_stress` is a per-rack float that
    the caller decays over time; when it's high, PSU hazard multiplies.
    """
    out: list[FailureEvent] = []
    base_hazard = 1.0 / (60 * 24 * ticks_per_hour)  # ~1 per rack per 60 days
    racks = {n.rack_id for n in fleet.nodes}
    for rack in racks:
        stress = (rack_thermal_stress or {}).get(rack, 0.0)
        hazard = base_hazard * (1.0 + 4.0 * stress)
        if rng.random() < hazard:
            parent_id = f"psu-{rack}-{tick}"
            out.append(
                FailureEvent(
                    tick=tick,
                    kind=FailureKind.PSU_TRIP,
                    scope="rack",
                    target=rack,
                    severity="critical",
                    benign=False,
                    parent_event_id=parent_id,
                )
            )
            # cascade: every node in the rack goes unreachable
            for node in fleet.by_rack(rack):
                out.append(
                    FailureEvent(
                        tick=tick,
                        kind=FailureKind.PSU_TRIP,
                        scope="node",
                        target=node.node_id,
                        severity="critical",
                        benign=False,
                        parent_event_id=parent_id,
                    )
                )
    return out


def network_flap(
    fleet: Fleet, tick: int, rng: random.Random, ticks_per_hour: int = 12
) -> list[FailureEvent]:
    """Spine-scope: many nodes show comm errors when a spine switch flaps."""
    out: list[FailureEvent] = []
    hazard = 1.0 / (7 * 24 * ticks_per_hour)  # ~1 per spine per week
    spines = {n.spine_id for n in fleet.nodes}
    for spine in spines:
        if rng.random() < hazard:
            parent_id = f"netflap-{spine}-{tick}"
            out.append(
                FailureEvent(
                    tick=tick,
                    kind=FailureKind.NETWORK_FLAP,
                    scope="spine",
                    target=spine,
                    severity="warn",
                    benign=False,
                    parent_event_id=parent_id,
                )
            )
            # ~30% of nodes on this spine see comm errors
            for node in fleet.nodes:
                if node.spine_id == spine and rng.random() < 0.3:
                    out.append(
                        FailureEvent(
                            tick=tick,
                            kind=FailureKind.NETWORK_FLAP,
                            scope="node",
                            target=node.node_id,
                            severity="warn",
                            benign=False,
                            parent_event_id=parent_id,
                        )
                    )
    return out


def power_event(
    fleet: Fleet, tick: int, rng: random.Random, ticks_per_hour: int = 12
) -> list[FailureEvent]:
    """Feed-scope: every rack on the feed loses power. Rare but catastrophic."""
    out: list[FailureEvent] = []
    hazard = 1.0 / (90 * 24 * ticks_per_hour)  # ~1 per feed per quarter
    feeds = {n.feed_id for n in fleet.nodes}
    for feed in feeds:
        if rng.random() < hazard:
            parent_id = f"feed-{feed}-{tick}"
            out.append(
                FailureEvent(
                    tick=tick,
                    kind=FailureKind.POWER_EVENT,
                    scope="feed",
                    target=feed,
                    severity="critical",
                    benign=False,
                    parent_event_id=parent_id,
                )
            )
            for node in fleet.by_feed(feed):
                out.append(
                    FailureEvent(
                        tick=tick,
                        kind=FailureKind.POWER_EVENT,
                        scope="node",
                        target=node.node_id,
                        severity="critical",
                        benign=False,
                        parent_event_id=parent_id,
                    )
                )
    return out
