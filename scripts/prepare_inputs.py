"""Publish the de-identified telemetry stream from the raw two-hour export.

Run once, privately, with the raw export and the de-identification key of the foundation project:

    TDF_SRC=/path/to/telematics-data-foundation/src \\
    TDF_DEID_KEY_FILE=/path/to/deid_key.txt \\
    python scripts/prepare_inputs.py --raw raw/eld_record.csv

The raw export never enters the repository. What is published:

* ``data/input/telemetry_2026-09-28.parquet`` - one row per device report, identifiers replaced by
  the foundation project's keyed pseudonyms (``DEV-``, ``VIN-``, ``DRV-``, same key and namespaces, so
  every row joins to ``dim_device``, ``dim_vehicle`` and the ECM-disconnect project's device table),
  coordinates coarsened to a 0.05 degree grid, IMEI / ICCID / record UUIDs / device-internal
  sequence numbers dropped, numbers parsed, timestamps converted to UTC with the device's offset kept
  in its own column. Columns that are empty in the whole export (diagnostic trouble codes, pedal and
  transmission fields) are dropped and listed in SOURCE.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("TDF_SRC", ""))
from tdf.deid import Pseudonymizer, coarsen_coordinate, to_number  # noqa: E402

PLACEHOLDER_VIN = {"VINNOTYETDETECTED", "VIN NOT YET DETECTED", "", "-"}
PLACEHOLDER_DRIVER = {"NODRIVERID", "", "-"}
PLACEHOLDER_TRAILER = {"NOTRAILERGPSID", "", "-"}
EMPTY_COLUMNS = ["Diagnostic Trouble Code", "Accelerator Pedal Position", "Brake Pedal Switch", "Transmission Health"]


def org_group(series: pd.Series) -> pd.Series:
    """'Vendor / Site / RT64' -> 'RT64' (the vendor and site names are dropped), as in the foundation."""
    return series.astype(str).str.split("/").str[-1].str.strip().replace({"nan": None, "-": None})


def to_pct(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.rstrip("%").replace({"nan": None, "-": None}), errors="coerce")


def build(raw: pd.DataFrame, p: Pseudonymizer) -> pd.DataFrame:
    d = pd.DataFrame()
    d["device_key"] = p.series(raw["Serial Number"], "device", "DEV-")
    d["org_group"] = org_group(raw["Organization"])
    d["is_internal_test"] = d["org_group"].fillna("").str.contains("test", case=False)
    stamp = pd.to_datetime(raw["Time Stamp"], utc=True, errors="coerce")
    d["ts"] = stamp
    d["tz_offset"] = raw["Time Zone"].replace("-", np.nan)
    d["terminal_timezone"] = raw["Terminal Timezone"]
    d["mode"] = raw["Mode"].replace("-", np.nan)
    d["event_type"] = raw["Event Type"]
    d["driver_key"] = p.series(raw["Driver ID"], "driver_uid", "DRV-", normalize=lambda x: str(x).strip(), placeholders=PLACEHOLDER_DRIVER)
    d["vin_key"] = p.series(raw["VIN"], "vin", "VIN-", placeholders=PLACEHOLDER_VIN)
    d["is_placeholder_vin"] = raw["VIN"].astype(str).str.strip().str.upper().isin(PLACEHOLDER_VIN)
    d["speed_kmh"] = to_number(raw["Speed"])
    d["rpm"] = to_number(raw["RPM"])
    d["engine_hours"] = to_number(raw["Engine Hours"])
    d["engine_miles"] = to_number(raw["Engine Miles"])
    d["gps_speed_kmh"] = to_number(raw["GPS Speed"])
    d["gps_valid"] = raw["GPS Valid"].map({"Yes": True, "No": False})
    d["lat_coarse"] = raw["Latitude"].map(coarsen_coordinate)
    d["lon_coarse"] = raw["Longitude"].map(coarsen_coordinate)
    d["is_stored"] = pd.to_numeric(raw["Is Stored"], errors="coerce").fillna(0).astype(int).astype(bool)
    d["is_compensated"] = pd.to_numeric(raw["Is Compensated"], errors="coerce").map({0: False, 1: True})
    d["car_battery_connected"] = raw["Car Battery Voltage"].map({"Yes": True, "No": False})
    d["backup_battery_pct"] = to_pct(raw["Backup Battery"])
    d["device_temp_c"] = to_number(raw["Device Temperature"])
    d["disk_used_pct"] = to_pct(raw["Disk Size"])
    d["has_trailer_gps"] = ~raw["Trailer GPS ID"].astype(str).str.strip().str.upper().isin(PLACEHOLDER_TRAILER | {"NAN"})
    d["hw_version"] = raw["ELD Hardware Version"].replace("-", np.nan)
    d["fw_version"] = raw["ELD Firmware Version"].replace("-", np.nan)
    d["st_fw_version"] = raw["ST Firmware Version"].replace("-", np.nan)
    d["ble_fw_version"] = raw["BLE Firmware Version"].replace("-", np.nan)
    d["model_code"] = raw["Model Code"].replace("-", np.nan)
    d["source_address"] = raw["Source Address"].replace("-", np.nan)
    d["source_address_lists"] = raw["Source Address Lists"].replace("-", np.nan)
    d = d.sort_values(["ts", "device_key"], kind="mergesort").reset_index(drop=True)
    d["row_id"] = np.arange(len(d), dtype=np.int64)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/input/telemetry_2026-09-28.parquet"))
    a = ap.parse_args(argv)
    raw = pd.read_csv(a.raw, dtype=str, low_memory=False, encoding="utf-8-sig")
    for col in EMPTY_COLUMNS:
        assert raw[col].isna().all(), f"{col} is not empty any more; review before publishing"
    p = Pseudonymizer.from_env()
    d = build(raw, p)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out, index=False, compression="zstd")
    print(f"{len(d)} rows, {d['device_key'].nunique()} devices, {d['ts'].min()} -> {d['ts'].max()}, {a.out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
