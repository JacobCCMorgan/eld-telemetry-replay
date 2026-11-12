import pandas as pd

from eldreplay import replay, trips

T0 = pd.Timestamp("2026-09-28 17:00:00", tz="UTC")


def make(rows):
    """rows: (sec, gps, speed[, miles[, mode]]) for device A unless a dict with dev is given."""
    recs = []
    for i, r in enumerate(rows):
        sec, gps, speed = r[:3]
        miles = r[3] if len(r) > 3 else None
        mode = r[4] if len(r) > 4 else "Driving"
        recs.append({"row_id": i, "device_key": "A", "ts": T0 + pd.Timedelta(seconds=sec), "mode": mode, "speed_kmh": speed, "rpm": 1200.0 if mode == "Driving" else None,
                     "engine_hours": 1.0, "engine_miles": miles, "gps_speed_kmh": gps, "gps_valid": True, "is_stored": False, "lat_coarse": 34.05, "lon_coarse": -118.25})
    return pd.DataFrame(recs)


def run(rows):
    seg = trips.segment(replay.to_reports(make(rows)))
    return seg.trips(), seg.stops()


def test_two_trips_separated_by_a_long_stop():
    rows = [(0, 0, 0, 100.0)]                                                  # parked at the first report, so the first trip is not truncated
    rows += [(s, 60, 60, 100 + s / 60) for s in range(30, 600, 30)]           # moving until 09:30
    rows += [(s, 0, 0, 110.0) for s in range(600, 1200, 30)]                   # 10 min stationary -> stop
    rows += [(s, 60, 60, 110 + (s - 1200) / 60) for s in range(1200, 1500, 30)]  # 5 min moving, open at the end
    t, st = run(rows)
    assert len(t) == 2
    a, b = t.iloc[0], t.iloc[1]
    assert a["start_ts"] == T0 + pd.Timedelta(seconds=30) and a["end_ts"] == T0 + pd.Timedelta(seconds=600) and a["end_reason"] == "stop"
    assert not a["truncated_start"] and not a["truncated_end"]
    assert a["distance_miles"] == 9.5 and a["n_reports"] == 30           # odometer 100.5 -> 110; 19 moving reports + the stationary ones until the stop is confirmed
    assert b["end_reason"] == "window" and b["truncated_end"] and b["start_ts"] == T0 + pd.Timedelta(seconds=1200)
    assert len(st) == 1 and st.iloc[0]["duration_s"] == 600 and st.iloc[0]["end_reason"] == "departure"


def test_short_stop_stays_inside_the_trip():
    rows = [(s, 60, 60) for s in range(0, 300, 30)] + [(s, 0, 0) for s in range(300, 480, 30)] + [(s, 60, 60) for s in range(480, 900, 30)] + [(900, 0, 0, None, "Sleeping")]
    t, st = run(rows)
    assert len(t) == 1 and t.iloc[0]["end_reason"] == "sleep" and t.iloc[0]["duration_s"] == 900
    assert len(st) == 0


def test_gap_ends_the_trip_at_the_last_report():
    rows = [(s, 60, 60) for s in range(0, 300, 30)] + [(s, 60, 60) for s in range(1200, 1500, 30)]
    t, _ = run(rows)
    assert len(t) == 2 and t.iloc[0]["end_reason"] == "gap" and t.iloc[0]["end_ts"] == T0 + pd.Timedelta(seconds=270)


def test_truncated_start_and_stop_at_window_end():
    rows = [(s, 60, 60) for s in range(0, 120, 30)] + [(s, 0, 0) for s in range(120, 900, 30)]
    t, st = run(rows)
    assert len(t) == 1 and t.iloc[0]["truncated_start"] and t.iloc[0]["end_reason"] == "stop"
    assert len(st) == 1 and st.iloc[0]["end_reason"] == "window" and st.iloc[0]["lat_coarse"] == 34.05


def test_published_stream_trip_counts_are_stable():
    s = replay.read_stream("data/input/telemetry_2026-09-28.parquet")
    seg = trips.segment(replay.to_reports(s))
    t = seg.trips()
    assert t["device_key"].nunique() > 1500 and len(t) > 1500
    assert (t["duration_s"] >= 0).all() and set(t["end_reason"]) <= {"stop", "sleep", "gap", "window"}
    assert t["distance_miles"].dropna().ge(-0.01).mean() > 0.99
