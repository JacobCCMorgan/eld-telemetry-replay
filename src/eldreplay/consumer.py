"""Stateful per-device consumer: derived state, ECM-disconnect rules, consistency checks.

Every report updates one ``DeviceTracker`` and produces one observation row. The checks are the
ones a backend needs in real time:

* ``moving``: GPS speed at or above the firmware's driving trigger (8 km/h, from the firmware log
  parser project) or a reported wheel speed at or above it;
* ``ecm_silent``: moving on GPS while the ECM delivers no RPM, the condition the ECM-disconnect
  project counts; ``ecm_silent_rpm_or_speed`` also accepts a missing wheel speed;
* ERR005 under two rules with the firmware's one-hour per-device cooldown: the shipped rule fires
  when GPS speed is positive and any of speed, RPM, engine hours or engine miles is missing; the
  corrected rule looks at speed and RPM only (the log parser showed engine hours is never available
  on OBD-II vehicles, so the shipped rule misfires there);
* ``speed_mismatch``: wheel speed and GPS speed both present and more than 15 km/h apart while
  either says the vehicle moves;
* ``odometer_regress`` / ``odometer_jump``: engine miles decreasing, or increasing faster than
  300 km/h would allow, between two consecutive reports of the same device;
* ``gap_s`` / ``gap_while_driving``: seconds since the device's previous report, flagged when a
  device that was driving goes quiet for more than 90 s (three missed 30-second reports);
* ``gps_invalid_moving``: wheel speed says moving while the GPS fix is flagged invalid;
* ``sleeping_while_moving``: reported mode Sleeping while GPS says moving.

The tracker keeps only what the next report needs (last timestamp, last odometer, last derived
state, cooldown deadlines), so memory is one small record per device.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .replay import Report

MOVING_KMH = 8.0
STATIONARY_KMH = 1.0
SPEED_MISMATCH_KMH = 15.0
ODOMETER_MAX_KMH = 300.0
GAP_S = 90.0
ERR005_COOLDOWN_S = 3600.0
ECM_FIELDS_SHIPPED = ("speed_kmh", "rpm", "engine_hours", "engine_miles")
ECM_FIELDS_CORRECTED = ("speed_kmh", "rpm")
COUNTED = ("ecm_silent", "err005_shipped_sent", "err005_corrected_sent", "speed_mismatch", "odometer_regress", "odometer_jump", "gap_while_driving", "gps_invalid_moving", "sleeping_while_moving")
EVENT_MODES = {"UnidentifiedDriving", "UnidentifiedDrivingDiagnostic", "UnidentifiedClearedSpeed", "UnidentifiedClearedDriver", "PreOta", "ExitOta", "LowBattery", "Shipping"}


def derived_state(r: Report) -> str:
    """driving (moving), idling (engine data but not moving), sleeping, event (non-telemetry record) or silent (no engine data, not moving)."""
    if r.mode in EVENT_MODES:
        return "event"
    if r.mode == "Sleeping":
        return "sleeping"
    gps = r.gps_speed_kmh or 0.0
    spd = r.speed_kmh or 0.0
    if gps >= MOVING_KMH or spd >= MOVING_KMH:
        return "driving"
    if (r.rpm or 0.0) > 0 or spd > 0:
        return "idling"
    if r.rpm is None and r.speed_kmh is None:
        return "silent"
    return "stationary"


@dataclass
class DeviceTracker:
    device_key: str
    last_ts: pd.Timestamp | None = None
    last_state: str | None = None
    last_miles: float | None = None
    err005_shipped_until: pd.Timestamp | None = None
    err005_corrected_until: pd.Timestamp | None = None
    reports: int = 0
    counts: dict = field(default_factory=dict)

    def _bump(self, key: str):
        self.counts[key] = self.counts.get(key, 0) + 1

    def observe(self, r: Report) -> dict:
        state = derived_state(r)
        gps = r.gps_speed_kmh
        spd = r.speed_kmh
        moving = state == "driving"
        gps_positive = (gps or 0.0) > 0
        missing_shipped = gps_positive and any(getattr(r, f) is None for f in ECM_FIELDS_SHIPPED)
        missing_corrected = gps_positive and any(getattr(r, f) is None for f in ECM_FIELDS_CORRECTED)
        gap = None if self.last_ts is None else float((r.ts - self.last_ts).total_seconds())
        o = {
            "row_id": r.row_id, "device_key": r.device_key, "ts": r.ts, "arrival_ts": r.arrival_ts, "mode": r.mode, "is_stored": r.is_stored,
            "state": state, "moving": moving,
            "ecm_silent": gps_positive and r.rpm is None,
            "ecm_silent_rpm_or_speed": gps_positive and (r.rpm is None or spd is None),
            "err005_shipped": missing_shipped, "err005_shipped_sent": False,
            "err005_corrected": missing_corrected, "err005_corrected_sent": False,
            "speed_mismatch": spd is not None and gps is not None and max(spd, gps) >= MOVING_KMH and abs(spd - gps) > SPEED_MISMATCH_KMH,
            "odometer_regress": False, "odometer_jump": False,
            "gap_s": gap, "gap_while_driving": gap is not None and gap > GAP_S and self.last_state == "driving",
            "gps_invalid_moving": r.gps_valid is False and (spd or 0.0) >= MOVING_KMH,
            "sleeping_while_moving": r.mode == "Sleeping" and (gps or 0.0) >= MOVING_KMH,
        }
        if missing_shipped and (self.err005_shipped_until is None or r.ts >= self.err005_shipped_until):
            o["err005_shipped_sent"] = True
            self.err005_shipped_until = r.ts + pd.Timedelta(seconds=ERR005_COOLDOWN_S)
        if missing_corrected and (self.err005_corrected_until is None or r.ts >= self.err005_corrected_until):
            o["err005_corrected_sent"] = True
            self.err005_corrected_until = r.ts + pd.Timedelta(seconds=ERR005_COOLDOWN_S)
        if r.engine_miles is not None and self.last_miles is not None and gap is not None and gap > 0:
            delta = r.engine_miles - self.last_miles
            if delta < -0.01:
                o["odometer_regress"] = True
            elif delta * 1.609344 / (gap / 3600.0) > ODOMETER_MAX_KMH:
                o["odometer_jump"] = True
        for k in COUNTED:
            if o[k]:
                self._bump(k)
        self._bump(f"state_{state}")
        self.reports += 1
        self.last_ts, self.last_state = r.ts, state
        if r.engine_miles is not None:
            self.last_miles = r.engine_miles
        return o


class FleetConsumer:
    """One tracker per device; collects observations and per-device summaries."""

    def __init__(self):
        self.trackers: dict[str, DeviceTracker] = {}
        self.rows: list[dict] = []

    def put(self, r: Report) -> None:
        t = self.trackers.get(r.device_key)
        if t is None:
            t = self.trackers[r.device_key] = DeviceTracker(r.device_key)
        self.rows.append(t.observe(r))

    def observations(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def device_summary(self) -> pd.DataFrame:
        recs = []
        for t in self.trackers.values():
            d = {"device_key": t.device_key, "reports": t.reports}
            d.update(t.counts)
            recs.append(d)
        return pd.DataFrame(recs).fillna(0)


def consume(reports: Iterable[Report]) -> FleetConsumer:
    c = FleetConsumer()
    for r in reports:
        c.put(r)
    return c
