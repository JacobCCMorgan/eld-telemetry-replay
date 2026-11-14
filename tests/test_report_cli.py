"""The report and the command line on a small synthetic stream (fixture only; the real numbers live in reports/)."""
import json

import pandas as pd
import pytest

from eldreplay import cli, report

T0 = pd.Timestamp("2026-09-28 17:00:00", tz="UTC")
GROUPS = {"A": "RT64", "B": "Test Group", "C": "1708", "D": "RT67"}


def _row(i, dev, sec, **kw):
    base = {"row_id": i, "device_key": dev, "org_group": GROUPS[dev], "is_internal_test": GROUPS[dev] == "Test Group", "ts": T0 + pd.Timedelta(seconds=sec), "mode": "Driving",
            "event_type": None, "driver_key": "DRV-1" if dev == "A" else None, "speed_kmh": 60.0, "rpm": 1300.0, "engine_hours": 10.0, "engine_miles": 1000.0 + sec / 60,
            "gps_speed_kmh": 60.0, "gps_valid": True, "is_stored": False, "lat_coarse": 34.05, "lon_coarse": -118.25}
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def stream():
    rows = []
    for k, sec in enumerate(range(0, 3600, 30)):
        rows.append(_row(len(rows), "A", sec, is_stored=1200 <= sec < 1800))                                  # A drives an hour, offline 20:00-29:30
        if sec < 1200:                                                                                        # B: ECM silent on GPS motion for 20 min, then sleeps
            rows.append(_row(len(rows), "B", sec, speed_kmh=None, rpm=None, engine_hours=None, engine_miles=None, gps_speed_kmh=70.0))
        else:
            rows.append(_row(len(rows), "B", sec, mode="Sleeping", speed_kmh=None, rpm=None, engine_hours=0.0, engine_miles=0.0, gps_speed_kmh=0.0))
        rows.append(_row(len(rows), "C", sec, engine_miles=5000.0 + (k % 3) * 40))                            # C: J1708 unit, odometer jumping around
        rows.append(_row(len(rows), "D", sec, speed_kmh=125.0, rpm=4016.0, gps_speed_kmh=0.0, engine_miles=138029.69))   # D: frozen ECM frame
    rows.append(_row(len(rows), "B", 3630, mode="UnidentifiedDriving", event_type="unidentified_driving", speed_kmh=None, rpm=None, engine_hours=None, engine_miles=None,
                     gps_speed_kmh=None, gps_valid=None))
    return pd.DataFrame(rows)


def test_report_end_to_end(stream, tmp_path):
    priority = pd.DataFrame({"device_key": ["B", "D", "Z"], "tier": ["P1", "P2", "P1"]})
    a = report.analyze(stream, priority, lateness_levels=(0, 300, float("inf")))
    assert a["flags"].set_index("check").loc["ecm_frozen", "devices"] == 1
    assert a["groups"].set_index("org_group").loc["1708", "odometer_regress"] > 0
    assert a["lateness"]["stored_reports"] == 20 and a["lateness"]["lateness_s_max"] == 620      # the 20:00 report arrives with the 30:00 live report plus 20 s
    assert a["tiers"].set_index("tier").loc["P1", "in_stream"] == 1
    path = report.write(stream, a, tmp_path / "out")
    text = path.read_text(encoding="ascii")
    assert text.startswith("# Replaying") and "## Findings" in text and "ecm_frozen" in text
    assert "## 3. The ECM-disconnect tiers on this stream" in text
    for f in ("01_fleet_timeline", "02_positions", "03_ecm_silence_by_batch", "04_priority_tiers_on_stream", "05_late_data", "06_trips_and_stops"):
        assert (tmp_path / "out" / "figures" / f"{f}.png").stat().st_size > 1000
    summary = json.loads((tmp_path / "out" / "tables" / "summary.json").read_text())
    assert summary["stream"]["devices"] == 4 and summary["checks"]["ecm_silent"]["devices"] == 1
    assert (tmp_path / "out" / "tables" / "trips.parquet").exists()


def test_report_without_priority_list(stream, tmp_path):
    a = report.analyze(stream, None, lateness_levels=(0, float("inf")))
    assert "tiers" not in a
    text = report.write(stream, a, tmp_path / "out").read_text(encoding="ascii")
    assert "## 3." not in text and "priority tiers" not in text


def test_cli_subcommands(stream, tmp_path, capsys):
    src = tmp_path / "s.parquet"
    stream.to_parquet(src, index=False)
    assert cli.main(["--input", str(src), "replay", "--out", str(tmp_path / "r.jsonl")]) == 0
    assert "replayed 481 reports from 4 devices" in capsys.readouterr().out
    assert sum(1 for _ in open(tmp_path / "r.jsonl")) == 481
    assert cli.main(["--input", str(src), "--devices", "2", "consume", "--out", str(tmp_path / "obs.parquet")]) == 0
    out = capsys.readouterr().out
    assert "ecm_silent" in out and pd.read_parquet(tmp_path / "obs.parquet")["device_key"].nunique() == 2
    assert cli.main(["--input", str(src), "trips", "--out", str(tmp_path / "t")]) == 0
    assert json.loads(capsys.readouterr().out)["trips"] >= 3
    assert (tmp_path / "t" / "trips.parquet").exists()
    assert cli.main(["--input", str(src), "latedata"]) == 0
    assert "arrival_order" in capsys.readouterr().out
    pri = tmp_path / "p.csv"
    pd.DataFrame({"device_key": ["B"], "tier": ["P1"]}).to_csv(pri, index=False)
    assert cli.main(["--input", str(src), "report", "--out", str(tmp_path / "rep"), "--priority", str(pri)]) == 0
    assert (tmp_path / "rep" / "replay_report.md").exists()
