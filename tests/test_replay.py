import time

import pandas as pd
import pytest

from eldreplay import replay

T0 = pd.Timestamp("2026-09-28 17:00:00", tz="UTC")


def frame(rows):
    """rows: (device, seconds, mode, speed, rpm, gps, stored[, arrival_seconds])"""
    recs = []
    for i, r in enumerate(rows):
        dev, sec, mode, speed, rpm, gps, stored = r[:7]
        arrival = r[7] if len(r) > 7 else sec
        recs.append({"row_id": i, "device_key": dev, "ts": T0 + pd.Timedelta(seconds=sec), "arrival_ts": T0 + pd.Timedelta(seconds=arrival), "mode": mode,
                     "speed_kmh": speed, "rpm": rpm, "engine_hours": 10.0, "engine_miles": 1000.0 + sec / 100, "gps_speed_kmh": gps, "gps_valid": True, "is_stored": stored, "org_group": "RT64"})
    return pd.DataFrame(recs)


def test_order_is_arrival_then_event_then_device():
    f = frame([("B", 60, "Driving", 50, 1200, 50, False), ("A", 60, "Driving", 50, 1200, 50, False), ("A", 0, "Driving", 50, 1200, 50, True, 90), ("A", 30, "Driving", 50, 1200, 50, False)])
    seq = [(r.device_key, int((r.ts - T0).total_seconds())) for r in replay.to_reports(f)]
    assert seq == [("A", 30), ("A", 60), ("B", 60), ("A", 0)]     # the stored report arrives at 90 s and comes last
    late = [r for r in replay.to_reports(f) if r.is_stored][0]
    assert late.lateness_s == 90


def test_replay_is_deterministic_and_counts():
    f = frame([("A", s, "Driving", 50, 1200, 50, False) for s in range(0, 300, 30)] + [("B", s, "Sleeping", None, None, None, False) for s in range(0, 300, 300)])
    a = [r.row_id for r in replay.Replayer(f)]
    b = [r.row_id for r in replay.Replayer(f.sample(frac=1, random_state=3))]     # shuffled input, same order out
    assert a == b and len(a) == 11
    q = replay.MemoryQueue()
    rp = replay.Replayer(f, sinks=[q])
    assert rp.run() == 11 and q.count == 11 and len(q) == 11
    first = q.get()
    assert first.row_id == a[0] and len(q) == 10


def test_missing_values_become_none():
    f = frame([("A", 0, "Sleeping", None, None, None, False)])
    r = next(replay.to_reports(f))
    assert r.speed_kmh is None and r.rpm is None and r.gps_speed_kmh is None and r.mode == "Sleeping"
    assert r.fields["org_group"] == "RT64"


def test_pacing_waits_for_event_time():
    f = frame([("A", 0, "Driving", 50, 1200, 50, False), ("A", 30, "Driving", 50, 1200, 50, False)])
    t = time.monotonic()
    replay.Replayer(f, speed=300).run()      # 30 s of telemetry at 300x = 0.1 s
    assert time.monotonic() - t >= 0.09


def test_jsonlines_sink(tmp_path):
    f = frame([("A", 0, "Driving", 50, 1200, 50, False)])
    sink = replay.JsonLinesSink(tmp_path / "out.jsonl")
    replay.Replayer(f, sinks=[sink]).run()
    sink.close()
    line = (tmp_path / "out.jsonl").read_text().strip()
    assert '"device_key": "A"' in line and '"is_stored": false' in line


@pytest.mark.parametrize("n", [1, 7])
def test_published_stream_round_trip(n):
    s = replay.read_stream("data/input/telemetry_2026-09-28.parquet").head(n * 1000)
    rows = list(replay.to_reports(s))
    assert len(rows) == n * 1000 and rows[0].ts <= rows[-1].ts
