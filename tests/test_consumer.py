import pandas as pd

from eldreplay import consumer, replay

T0 = pd.Timestamp("2026-09-28 17:00:00", tz="UTC")


def make(rows):
    """rows: dicts with sec and any of mode, speed, rpm, hours, miles, gps, valid, stored."""
    recs = []
    for i, r in enumerate(rows):
        recs.append({"row_id": i, "device_key": r.get("dev", "A"), "ts": T0 + pd.Timedelta(seconds=r["sec"]), "mode": r.get("mode", "Driving"), "speed_kmh": r.get("speed"), "rpm": r.get("rpm"),
                     "engine_hours": r.get("hours", 100.0), "engine_miles": r.get("miles"), "gps_speed_kmh": r.get("gps"), "gps_valid": r.get("valid", True), "is_stored": r.get("stored", False)})
    return pd.DataFrame(recs)


def run(rows):
    c = consumer.consume(replay.to_reports(make(rows)))
    return c.observations(), c


def test_states():
    obs, _ = run([
        dict(sec=0, speed=50, rpm=1200, gps=50),
        dict(sec=30, speed=0, rpm=700, gps=0),
        dict(sec=60, mode="Sleeping"),
        dict(sec=90, speed=None, rpm=None, gps=0),
        dict(sec=120, mode="UnidentifiedDriving"),
        dict(sec=150, speed=0, rpm=0, gps=0),
    ])
    assert obs["state"].tolist() == ["driving", "idling", "sleeping", "silent", "event", "stationary"]


def test_ecm_silence_and_err005_rules_with_cooldown():
    rows = [dict(sec=s, speed=None, rpm=None, hours=None, gps=80) for s in range(0, 7200, 30)]   # 4 hours of GPS motion, ECM silent
    obs, c = run(rows)
    assert obs["ecm_silent"].all() and obs["err005_shipped"].all() and obs["err005_corrected"].all()
    assert obs["err005_shipped_sent"].sum() == 2 and obs["err005_corrected_sent"].sum() == 2    # one event per hour per rule
    assert c.trackers["A"].counts["ecm_silent"] == 240
    # OBD-II pattern: speed and RPM present, engine hours never -> shipped rule fires, corrected rule does not
    obs, _ = run([dict(sec=s, speed=60, rpm=1500, hours=None, gps=60) for s in range(0, 300, 30)])
    assert obs["err005_shipped_sent"].sum() == 1 and obs["err005_corrected"].sum() == 0 and obs["ecm_silent"].sum() == 0


def test_speed_mismatch_and_gps_invalid():
    obs, _ = run([dict(sec=0, speed=90, rpm=1400, gps=60), dict(sec=30, speed=90, rpm=1400, gps=85), dict(sec=60, speed=90, rpm=1400, gps=90, valid=False)])
    assert obs["speed_mismatch"].tolist() == [True, False, False]
    assert obs["gps_invalid_moving"].tolist() == [False, False, True]


def test_odometer_regress_and_jump():
    obs, _ = run([dict(sec=0, speed=50, rpm=1200, gps=50, miles=1000.0), dict(sec=30, speed=50, rpm=1200, gps=50, miles=1000.4),
                  dict(sec=60, speed=50, rpm=1200, gps=50, miles=999.9), dict(sec=90, speed=50, rpm=1200, gps=50, miles=1010.0)])
    assert obs["odometer_regress"].tolist() == [False, False, True, False]
    assert obs["odometer_jump"].tolist() == [False, False, False, True]      # 10 miles in 30 s
    # a zero reading is skipped, not a regression; the next valid reading is compared with the last valid one
    obs, _ = run([dict(sec=0, speed=50, rpm=1200, gps=50, miles=1000.0), dict(sec=30, speed=50, rpm=1200, gps=50, miles=0.0), dict(sec=60, speed=50, rpm=1200, gps=50, miles=1000.8),
                  dict(sec=90, mode="Sleeping", miles=0.0)])
    assert obs["odometer_regress"].sum() == 0 and obs["odometer_jump"].sum() == 0
    assert obs["odometer_zero"].tolist() == [False, True, False, False]


def test_gap_while_driving_and_sleeping_while_moving():
    obs, _ = run([dict(sec=0, speed=50, rpm=1200, gps=50), dict(sec=200, speed=50, rpm=1200, gps=50), dict(sec=230, mode="Sleeping", gps=40), dict(sec=600, mode="Sleeping")])
    assert obs["gap_s"].isna().tolist() == [True, False, False, False] and obs["gap_s"].tolist()[1:] == [200.0, 30.0, 370.0]
    assert obs["gap_while_driving"].tolist() == [False, True, False, False]
    assert obs["sleeping_while_moving"].tolist() == [False, False, True, False]


def test_devices_are_tracked_independently():
    obs, c = run([dict(dev="A", sec=0, speed=50, rpm=1200, gps=50, miles=10.0), dict(dev="B", sec=0, speed=50, rpm=1200, gps=50, miles=500.0),
                  dict(dev="A", sec=30, speed=50, rpm=1200, gps=50, miles=10.4), dict(dev="B", sec=30, speed=50, rpm=1200, gps=50, miles=499.0)])
    assert obs.set_index(["device_key", "row_id"])["odometer_regress"].tolist() == [False, False, False, True]
    s = c.device_summary().set_index("device_key")
    assert s.loc["A", "reports"] == 2 and s.loc["B", "odometer_regress"] == 1 and s.loc["A", "state_driving"] == 2
