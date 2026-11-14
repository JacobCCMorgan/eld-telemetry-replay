"""Command line: ``eldreplay replay | consume | trips | latedata | report``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import consumer, latedata, metrics, replay, report, trips

INPUT = "data/input/telemetry_2026-09-28.parquet"
PRIORITY = "data/input/ecm_priority_list_21def15.csv"


def _load(a):
    s = replay.read_stream(a.input)
    if getattr(a, "devices", None):
        keep = s["device_key"].drop_duplicates().head(a.devices)
        s = s[s["device_key"].isin(keep)]
    return s


def cmd_replay(a):
    s = _load(a)
    sinks = [replay.JsonLinesSink(Path(a.out))] if a.out else []
    n = replay.Replayer(s, sinks=sinks, speed=a.speed).run()
    for k in sinks:
        k.close()
    print(f"replayed {n:,} reports from {s['device_key'].nunique():,} devices" + (f" to {a.out}" if a.out else ""))
    return 0


def cmd_consume(a):
    s = _load(a)
    c = consumer.consume(replay.to_reports(s))
    obs = c.observations()
    print(metrics.flag_summary(obs).to_string(index=False))
    print(json.dumps(metrics.cadence_summary(obs), indent=2))
    if a.out:
        obs.to_parquet(a.out, index=False)
        print(f"observations written to {a.out}")
    return 0


def cmd_trips(a):
    s = _load(a)
    seg = trips.segment(replay.to_reports(s))
    t, st = seg.trips(), seg.stops()
    print(json.dumps(metrics.trip_summary(t, st, s), indent=2))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        t.to_parquet(Path(a.out) / "trips.parquet", index=False)
        st.to_parquet(Path(a.out) / "stops.parquet", index=False)
    return 0


def cmd_latedata(a):
    s = latedata.arrival_times(_load(a))
    print(json.dumps(latedata.lateness_summary(s), indent=2))
    print(latedata.run_strategies(s).to_string(index=False))
    return 0


def cmd_report(a):
    s = _load(a)
    priority = pd.read_csv(a.priority) if a.priority else None
    res = report.analyze(s, priority)
    path = report.write(s, res, Path(a.out))
    print(f"report written to {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eldreplay", description="replay 30-second ELD telemetry and run stateful consumers on it")
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--devices", type=int, default=None, help="limit to the first N devices (quick runs)")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay", help="push the stream through the replayer, optionally paced, optionally to a JSON-lines file")
    p.add_argument("--speed", type=float, default=None, help="event seconds per wall-clock second; omit for as fast as possible")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("consume", help="run the per-device consumer and print the check summary")
    p.add_argument("--out", default=None, help="write the observations as Parquet")
    p.set_defaults(fn=cmd_consume)
    p = sub.add_parser("trips", help="segment trips and stops")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_trips)
    p = sub.add_parser("latedata", help="arrival model and watermark strategies")
    p.set_defaults(fn=cmd_latedata)
    p = sub.add_parser("report", help="everything: figures, tables and the Markdown report")
    p.add_argument("--out", default="reports")
    p.add_argument("--priority", default=PRIORITY, help="the ECM-disconnect project's priority list, or '' to skip the tier check")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
