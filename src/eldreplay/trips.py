"""Trip and stop segmentation as a streaming consumer.

A trip starts with the first report that says the vehicle moves (GPS or wheel speed at or above
8 km/h) and ends when the vehicle has been stationary for at least five minutes (the trip ended at
the start of that stationary span), when the device reports Sleeping, or when no report arrives for
ten minutes (the trip ended at the last report). The stationary spans between trips that last at
least five minutes are the stops; shorter ones (traffic lights, a gate) stay inside the trip.

Trips that are already in motion at the device's first report are flagged ``truncated_start``;
trips still open when the stream ends are closed at the last report and flagged ``truncated_end``.
Distance is the odometer difference between the last valid readings at both ends, where both exist.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .consumer import MOVING_KMH, STATIONARY_KMH
from .replay import Report

STOP_MIN_S = 300.0
GAP_END_S = 600.0


def is_moving(r: Report) -> bool:
    return (r.gps_speed_kmh or 0.0) >= MOVING_KMH or (r.speed_kmh or 0.0) >= MOVING_KMH


def is_stationary(r: Report) -> bool:
    return (r.gps_speed_kmh or 0.0) < STATIONARY_KMH and (r.speed_kmh or 0.0) < STATIONARY_KMH


def _miles(r: Report) -> float | None:
    return r.engine_miles if r.engine_miles is not None and r.engine_miles > 0 else None


@dataclass
class _Trip:
    device_key: str
    start_ts: pd.Timestamp
    start_miles: float | None
    truncated_start: bool
    last_ts: pd.Timestamp
    last_miles: float | None = None
    n_reports: int = 0
    max_gps_kmh: float = 0.0
    stationary_since: pd.Timestamp | None = None
    stationary_lat: float | None = None
    stationary_lon: float | None = None

    def close(self, end_ts: pd.Timestamp, reason: str, truncated_end: bool = False) -> dict:
        dist = None if self.start_miles is None or self.last_miles is None else round(self.last_miles - self.start_miles, 2)
        return {"device_key": self.device_key, "start_ts": self.start_ts, "end_ts": end_ts, "duration_s": float((end_ts - self.start_ts).total_seconds()), "n_reports": self.n_reports,
                "max_gps_kmh": self.max_gps_kmh, "distance_miles": dist, "end_reason": reason, "truncated_start": self.truncated_start, "truncated_end": truncated_end}


@dataclass
class DeviceTrips:
    device_key: str
    trip: _Trip | None = None
    first_report: bool = True
    last_ts: pd.Timestamp | None = None
    stop_since: pd.Timestamp | None = None     # stationary span outside a trip
    stop_lat: float | None = None
    stop_lon: float | None = None
    trips: list[dict] = field(default_factory=list)
    stops: list[dict] = field(default_factory=list)

    def _end_stop(self, end_ts: pd.Timestamp, reason: str):
        if self.stop_since is not None:
            dur = float((end_ts - self.stop_since).total_seconds())
            if dur >= STOP_MIN_S:
                self.stops.append({"device_key": self.device_key, "start_ts": self.stop_since, "end_ts": end_ts, "duration_s": dur,
                                   "lat_coarse": self.stop_lat, "lon_coarse": self.stop_lon, "end_reason": reason})
            self.stop_since = None

    def put(self, r: Report) -> None:
        moving, stationary = is_moving(r), is_stationary(r)
        lat, lon = r.fields.get("lat_coarse"), r.fields.get("lon_coarse")
        if self.trip is not None and self.last_ts is not None and (r.ts - self.last_ts).total_seconds() > GAP_END_S:
            self.trips.append(self.trip.close(self.last_ts, "gap"))
            self.trip = None
        if self.trip is None:
            if moving:
                self._end_stop(r.ts, "departure")
                self.trip = _Trip(self.device_key, r.ts, _miles(r), self.first_report, r.ts)
            elif stationary and self.stop_since is None:
                self.stop_since, self.stop_lat, self.stop_lon = r.ts, lat, lon
            elif not stationary:
                self._end_stop(r.ts, "moving")
        if self.trip is not None:
            t = self.trip
            t.n_reports += 1
            t.last_ts = r.ts
            if _miles(r) is not None:
                t.last_miles = _miles(r)
            t.max_gps_kmh = max(t.max_gps_kmh, r.gps_speed_kmh or 0.0)
            if r.mode == "Sleeping":
                self.trips.append(t.close(r.ts, "sleep"))
                self.trip = None
                self.stop_since, self.stop_lat, self.stop_lon = r.ts, lat, lon
            elif stationary:
                if t.stationary_since is None:
                    t.stationary_since, t.stationary_lat, t.stationary_lon = r.ts, lat, lon
                elif (r.ts - t.stationary_since).total_seconds() >= STOP_MIN_S:
                    self.trips.append(t.close(t.stationary_since, "stop"))
                    self.stop_since, self.stop_lat, self.stop_lon = t.stationary_since, t.stationary_lat, t.stationary_lon
                    self.trip = None
            elif moving:
                t.stationary_since = None
        self.first_report = False
        self.last_ts = r.ts

    def finish(self, end_ts: pd.Timestamp) -> None:
        if self.trip is not None:
            self.trips.append(self.trip.close(self.trip.last_ts, "window", truncated_end=True))
            self.trip = None
        if self.stop_since is not None:
            self._end_stop(end_ts, "window")


class TripSegmenter:
    def __init__(self):
        self.devices: dict[str, DeviceTrips] = {}
        self.last_ts: pd.Timestamp | None = None

    def put(self, r: Report) -> None:
        d = self.devices.get(r.device_key)
        if d is None:
            d = self.devices[r.device_key] = DeviceTrips(r.device_key)
        d.put(r)
        self.last_ts = r.ts if self.last_ts is None else max(self.last_ts, r.ts)

    def finish(self) -> None:
        for d in self.devices.values():
            d.finish(self.last_ts)

    def trips(self) -> pd.DataFrame:
        cols = ["device_key", "start_ts", "end_ts", "duration_s", "n_reports", "max_gps_kmh", "distance_miles", "end_reason", "truncated_start", "truncated_end"]
        rows = [t for d in self.devices.values() for t in d.trips]
        return pd.DataFrame(rows, columns=cols).sort_values(["device_key", "start_ts"]).reset_index(drop=True)

    def stops(self) -> pd.DataFrame:
        cols = ["device_key", "start_ts", "end_ts", "duration_s", "lat_coarse", "lon_coarse", "end_reason"]
        rows = [s for d in self.devices.values() for s in d.stops]
        return pd.DataFrame(rows, columns=cols).sort_values(["device_key", "start_ts"]).reset_index(drop=True)


def segment(reports: Iterable[Report]) -> TripSegmenter:
    seg = TripSegmenter()
    for r in reports:
        seg.put(r)
    seg.finish()
    return seg
