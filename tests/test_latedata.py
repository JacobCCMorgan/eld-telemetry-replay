import math

import pandas as pd

from eldreplay import latedata

T0 = pd.Timestamp("2026-09-28 17:00:00", tz="UTC")


def make(rows):
    """rows: (dev, sec, gps, stored)"""
    recs = []
    for i, (dev, sec, gps, stored) in enumerate(rows):
        recs.append({"row_id": i, "device_key": dev, "ts": T0 + pd.Timedelta(seconds=sec), "mode": "Driving", "speed_kmh": gps, "rpm": 1200.0, "engine_hours": 1.0,
                     "engine_miles": 100.0 + sec / 60, "gps_speed_kmh": gps, "gps_valid": True, "is_stored": stored, "lat_coarse": 34.05, "lon_coarse": -118.25})
    return pd.DataFrame(recs)


def test_arrival_model():
    f = make([("A", 0, 60, False), ("A", 30, 60, True), ("A", 60, 60, True), ("A", 300, 60, False), ("A", 330, 60, True), ("B", 0, 60, False)])
    a = latedata.arrival_times(f).set_index("row_id")
    assert a.loc[1, "lateness_s"] == 270 and a.loc[2, "lateness_s"] == 240       # both arrive with the live report at 300 s
    assert a.loc[0, "lateness_s"] == 0 and a.loc[3, "lateness_s"] == 0
    assert a.loc[4, "arrival_after_window"] and a.loc[4, "lateness_s"] == 1      # no later live report: arrives after the window
    s = latedata.lateness_summary(a.reset_index())
    assert s["stored_reports"] == 3 and s["devices_with_stored"] == 1 and s["offline_episodes"] == 2 and s["arrive_after_window"] == 1
    assert s["lateness_s_max"] == 270


def test_dropping_late_records_splits_a_trip():
    # device drives 20 minutes; reports from 05:00 to 14:30 are stored and arrive with the live report at 14:30
    rows = [("A", s, 60, 300 <= s < 870) for s in range(0, 1200, 30)]
    a = latedata.arrival_times(make(rows))
    res = latedata.run_strategies(a, levels=(0, 60, 900, math.inf)).set_index(["strategy", "allowed_lateness_s"])
    oracle = res.loc[("oracle", None)]
    assert oracle["trips"] == 1 and oracle["records_dropped"] == 0
    drop0 = res.loc[("drop_late", 0)]
    assert drop0["records_dropped"] == 19 and drop0["trips"] == 1 and drop0["trips_ended_by_gap"] == 0     # the hole is 04:30 -> 14:30, exactly the gap limit, so the trip survives
    full = res.loc[("drop_late", 900)]
    assert full["records_dropped"] == 0 and full["devices_with_different_trip_count"] == 0
    # one more stored report makes the hole 10.5 minutes and the trip splits
    rows = [("A", s, 60, 300 <= s < 900) for s in range(0, 1200, 30)]
    res = latedata.run_strategies(latedata.arrival_times(make(rows)), levels=(0,)).set_index("strategy")
    assert res.loc["drop_late", "trips"] == 2 and res.loc["drop_late", "trips_ended_by_gap"] == 1 and res.loc["drop_late", "devices_with_different_trip_count"] == 1


def test_arrival_order_consumer_sees_time_going_backwards():
    rows = [("A", s, 60, 300 <= s < 900) for s in range(0, 1200, 30)]
    res = latedata.run_strategies(latedata.arrival_times(make(rows)), levels=()).set_index("strategy")
    assert res.loc["arrival_order", "records_dropped"] == 0
    assert res.loc["arrival_order", "trips"] >= 1
