"""Replay a telemetry table as an ordered event stream.

The replayer reads the published Parquet (or any frame with the same columns), orders the reports
by their arrival time and hands them one at a time to whoever consumes them. Arrival time defaults
to the event time (``ts``); the late-data experiments pass an ``arrival_ts`` column instead so the
stream is replayed the way a backend would have seen it, with offline records surfacing late.

Ordering is total and deterministic: (arrival time, event time, device, row_id). Replaying the same
table twice yields the same sequence, which the tests check.

Pacing is optional: ``speed=None`` replays as fast as the consumer accepts, ``speed=60`` plays one
minute of telemetry per second of wall-clock time. Sinks are pluggable through ``Sink``; the in-memory
queue is what the pipeline uses, a JSON-lines sink is there for debugging, and a broker (MQTT, Kafka)
is a ``Sink`` with a ``put``.
"""
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

ORDER = ["arrival_ts", "ts", "device_key", "row_id"]


@dataclass(frozen=True)
class Report:
    """One device report as the consumers see it. ``fields`` carries the remaining columns."""

    row_id: int
    device_key: str
    ts: pd.Timestamp
    arrival_ts: pd.Timestamp
    mode: str | None
    speed_kmh: float | None
    rpm: float | None
    engine_hours: float | None
    engine_miles: float | None
    gps_speed_kmh: float | None
    gps_valid: bool | None
    is_stored: bool
    fields: dict

    @property
    def lateness_s(self) -> float:
        return float((self.arrival_ts - self.ts).total_seconds())


class Sink(Protocol):
    def put(self, report: Report) -> None: ...


class MemoryQueue:
    """In-memory FIFO; ``get`` returns None when empty."""

    def __init__(self):
        self._q: deque[Report] = deque()
        self.count = 0

    def put(self, report: Report) -> None:
        self._q.append(report)
        self.count += 1

    def get(self) -> Report | None:
        return self._q.popleft() if self._q else None

    def __len__(self) -> int:
        return len(self._q)


class JsonLinesSink:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "w", encoding="ascii")

    def put(self, report: Report) -> None:
        d = {"row_id": report.row_id, "device_key": report.device_key, "ts": report.ts.isoformat(), "arrival_ts": report.arrival_ts.isoformat(), "mode": report.mode,
             "speed_kmh": report.speed_kmh, "rpm": report.rpm, "gps_speed_kmh": report.gps_speed_kmh, "is_stored": report.is_stored}
        self._fh.write(json.dumps(d) + "\n")

    def close(self) -> None:
        self._fh.close()


def _opt(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
    except TypeError:
        pass
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _opt_float(v) -> float | None:
    v = _opt(v)
    return None if v is None else float(v)


def ordered(stream: pd.DataFrame) -> pd.DataFrame:
    """The stream in replay order. Adds ``arrival_ts`` = ``ts`` when the frame has none."""
    s = stream
    if "arrival_ts" not in s.columns:
        s = s.assign(arrival_ts=s["ts"])
    if "row_id" not in s.columns:
        s = s.assign(row_id=np.arange(len(s)))
    return s.sort_values(ORDER, kind="mergesort").reset_index(drop=True)


def to_reports(stream: pd.DataFrame) -> Iterator[Report]:
    s = ordered(stream)
    core = {"row_id", "device_key", "ts", "arrival_ts", "mode", "speed_kmh", "rpm", "engine_hours", "engine_miles", "gps_speed_kmh", "gps_valid", "is_stored"}
    extra = [c for c in s.columns if c not in core]
    for rec in s.to_dict("records"):
        yield Report(
            row_id=int(rec["row_id"]), device_key=rec["device_key"], ts=rec["ts"], arrival_ts=rec["arrival_ts"], mode=_opt(rec.get("mode")),
            speed_kmh=_opt_float(rec.get("speed_kmh")), rpm=_opt_float(rec.get("rpm")), engine_hours=_opt_float(rec.get("engine_hours")),
            engine_miles=_opt_float(rec.get("engine_miles")), gps_speed_kmh=_opt_float(rec.get("gps_speed_kmh")), gps_valid=_opt(rec.get("gps_valid")),
            is_stored=bool(rec.get("is_stored", False)), fields={k: _opt(rec[k]) for k in extra},
        )


class Replayer:
    """Push the reports of a table into one or more sinks, in arrival order, optionally paced."""

    def __init__(self, stream: pd.DataFrame, sinks: Iterable[Sink] = (), speed: float | None = None):
        self.stream = stream
        self.sinks = list(sinks)
        self.speed = speed
        self.sent = 0

    def __iter__(self) -> Iterator[Report]:
        t0_wall = time.monotonic()
        t0_evt: pd.Timestamp | None = None
        for r in to_reports(self.stream):
            if self.speed:
                if t0_evt is None:
                    t0_evt = r.arrival_ts
                due = t0_wall + (r.arrival_ts - t0_evt).total_seconds() / self.speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            for s in self.sinks:
                s.put(r)
            self.sent += 1
            yield r

    def run(self) -> int:
        for _ in self:
            pass
        return self.sent


def read_stream(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)
