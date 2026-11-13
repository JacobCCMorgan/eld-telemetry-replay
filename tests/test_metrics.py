import pandas as pd

from eldreplay import consumer, metrics, replay, trips

T0 = pd.Timestamp("2026-09-28 17:00:00", tz="UTC")


def stream():
    rows = []
    i = 0
    for dev, grp, rpm in (("A", "RT64", 1200.0), ("B", "Test Group", None)):
        for sec in range(0, 1200, 30):
            rows.append({"row_id": i, "device_key": dev, "org_group": grp, "is_internal_test": grp == "Test Group", "ts": T0 + pd.Timedelta(seconds=sec), "mode": "Driving",
                         "event_type": None, "driver_key": "DRV-1" if dev == "A" else None, "speed_kmh": 60.0, "rpm": rpm, "engine_hours": 10.0, "engine_miles": 1000.0 + sec / 60,
                         "gps_speed_kmh": 60.0, "gps_valid": True, "is_stored": False})
            i += 1
    rows.append({"row_id": i, "device_key": "B", "org_group": "Test Group", "is_internal_test": True, "ts": T0 + pd.Timedelta(seconds=1230), "mode": "UnidentifiedDriving",
                 "event_type": "unidentified_driving", "driver_key": None, "speed_kmh": None, "rpm": None, "engine_hours": None, "engine_miles": None, "gps_speed_kmh": None,
                 "gps_valid": None, "is_stored": False})
    return pd.DataFrame(rows)


def test_flag_group_and_timeline_summaries():
    s = stream()
    c = consumer.consume(replay.to_reports(s))
    obs = c.observations()
    flags = metrics.flag_summary(obs).set_index("check")
    assert flags.loc["ecm_silent", "reports"] == 40 and flags.loc["ecm_silent", "devices"] == 1
    assert flags.loc["err005_corrected_sent", "reports"] == 1
    g = metrics.group_summary(obs, s).set_index("org_group")
    assert g.loc["Test Group", "ecm_silent_share_of_driving"] == 1.0 and g.loc["RT64", "ecm_silent"] == 0
    assert g.loc["Test Group", "devices_ecm_silent"] == 1
    tl = metrics.fleet_timeline(obs)
    assert len(tl) == 21 and tl["driving"].iloc[0] == 2 and tl["event"].iloc[-1] == 1
    cad = metrics.cadence_summary(obs)
    assert cad["median_s"] == 30.0 and cad["gaps_over_90s_while_driving"] == 0


def test_unidentified_trip_and_tier_summaries():
    s = stream()
    u = metrics.unidentified_summary(s)
    assert u["devices_with_unidentified_driving_event"] == 1 and u["devices_moving_without_driver"] == 1
    assert u["share_of_moving_reports_without_driver"] == 0.5
    seg = trips.segment(replay.to_reports(s))
    t = metrics.trip_summary(seg.trips(), seg.stops(), s)
    assert t["trips"] == 2 and t["complete_trips"] == 0 and t["driving_to_sleeping_transitions"] == 0
    c = consumer.consume(replay.to_reports(s))
    pri = pd.DataFrame({"device_key": ["A", "B", "Z"], "tier": ["P4", "P1", "P1"]})
    tiers = metrics.tier_check(c.device_summary(), pri, s).set_index("tier")
    assert tiers.loc["P1", "devices"] == 2 and tiers.loc["P1", "in_stream"] == 1 and tiers.loc["P1", "ecm_silent_share_of_driving"] == 1.0
    assert tiers.loc["P4", "ecm_silent_share_of_driving"] == 0.0
