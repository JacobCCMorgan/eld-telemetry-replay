# Published input

`telemetry_2026-09-28.parquet` is a two-hour export of a fleet's raw ELD reports (the 30-second
telemetry every device sends while driving, 5-minute status while sleeping, plus event records), de-identified
and published as the stream this project replays. It is real operational data of a small ELD provider,
not a simulation; the numbers describe that fleet on that afternoon.

| | |
|---|---|
| Rows | 354,874 reports |
| Devices | 2,191 (every one joins to the foundation project's `dim_device`) |
| Window | 2026-09-28 16:59:57 to 18:59:57 UTC (the 30-second cadence means about 240 reports per driving device) |
| Modes | Driving 333,970; Sleeping 17,862; unidentified-driving events 2,648; Standby 189; OTA, LowBattery, Shipping 205 |
| Stored (offline, re-uploaded) reports | 7,464 (2.1 %) |
| Size | 7.5 MB, zstd-compressed Parquet |

## De-identification (same key and namespaces as the foundation project)

| raw column | published as | treatment |
|---|---|---|
| Serial Number, Device ID | `device_key` | keyed pseudonym `DEV-` + 10 hex (joins `dim_device`, `eld-ecm-disconnect`, `eld-device-log-parser`) |
| Organization | `org_group`, `is_internal_test` | vendor and site names dropped, device batch label kept (`RT64`, `Volvo 2026`, `OBD II`, `Test Group`, ...) |
| VIN | `vin_key`, `is_placeholder_vin` | keyed pseudonym `VIN-` + 10 hex; `VINNOTYETDETECTED` becomes null with the flag set |
| Driver ID (app uid) | `driver_key` | keyed pseudonym `DRV-` + 10 hex; `NODRIVERID` becomes null |
| Latitude, Longitude | `lat_coarse`, `lon_coarse` | rounded to a 0.05-degree grid (about 5 km) |
| Time Stamp, Time Zone | `ts` (UTC), `tz_offset` | converted to UTC; the device's offset kept as text |
| IMEI, ICCID, Trailer GPS ID, ELD Unit ID, USN, ELD Seq ID, ELD Log UUID, Event UUID, Engine KM | dropped | identifiers, device-internal sequence numbers or redundant units; `has_trailer_gps` keeps the yes/no |
| Diagnostic Trouble Code, Accelerator Pedal Position, Brake Pedal Switch, Transmission Health | dropped | empty in every row of the export |
| everything else | parsed | numbers parsed (`678,506.81` -> 678506.81), `Yes`/`No` and `0`/`1` to booleans, percentages to numbers |

Rows are sorted by `ts` then `device_key`; `row_id` is that order. The publishing script is
`scripts/prepare_inputs.py`; it needs the raw export and the foundation key, neither of which is in
the repository.

## `ecm_priority_list_21def15.csv`

The device priority list published by the ECM-disconnect project (`eld-ecm-disconnect`,
`reports/priority_list.csv` at commit `21def15`): 182 devices with their tier (P1 to P4), ranked
from three days of the platform's anomaly export in September 2026. Copied unchanged; the device
keys share the foundation key, so they join this stream's `device_key` directly. Used by
`eldreplay report` to measure the tiers on the raw stream, an independent data path.
