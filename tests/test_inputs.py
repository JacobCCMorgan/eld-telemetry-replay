"""Invariants of the published stream: shape, de-identification, joins to the foundation key space."""
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input" / "telemetry_2026-09-28.parquet"


@pytest.fixture(scope="module")
def stream():
    return pd.read_parquet(INPUT)


def test_shape_and_window(stream):
    assert len(stream) == 354874 and stream["device_key"].nunique() == 2191
    assert str(stream["ts"].dt.tz) == "UTC"
    assert stream["ts"].min().isoformat() == "2026-09-28T16:59:57+00:00" and stream["ts"].max().isoformat() == "2026-09-28T18:59:57+00:00"
    assert stream["ts"].is_monotonic_increasing and (stream["row_id"].values == range(len(stream))).all()


def test_pseudonymized(stream):
    assert stream["device_key"].str.fullmatch(r"DEV-[0-9a-f]{10}").all()
    assert stream["vin_key"].dropna().str.fullmatch(r"VIN-[0-9a-f]{10}").all()
    assert stream["driver_key"].dropna().str.fullmatch(r"DRV-[0-9a-f]{10}").all()
    for col in ("org_group", "terminal_timezone", "fw_version", "st_fw_version", "source_address"):
        text = " ".join(stream[col].dropna().unique())
        assert not re.search(r"RT6\d{7}|\b\d{15}\b|@", text), col
    assert "/" not in " ".join(stream["org_group"].dropna().unique())
    assert set(stream.columns).isdisjoint({"IMEI", "ICCID", "USN", "Latitude", "Longitude"})


def test_coordinates_on_grid(stream):
    for col in ("lat_coarse", "lon_coarse"):
        vals = stream[col].dropna()
        assert ((vals * 20).round() - vals * 20).abs().max() < 1e-6


def test_modes_and_flags(stream):
    counts = stream["mode"].value_counts()
    assert counts["Driving"] == 333970 and counts["Sleeping"] == 17862
    assert int(stream["is_stored"].sum()) == 7464
    assert stream.loc[stream["is_placeholder_vin"], "vin_key"].isna().all()
    assert stream.loc[stream["mode"] == "Driving", "source_address"].notna().mean() > 0.98
