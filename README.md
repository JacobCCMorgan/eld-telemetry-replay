# eld-telemetry-replay

Replay a fleet's 30-second ELD telemetry as an event stream and run stateful consumers on it:
per-device state reconstruction, trip and stop segmentation, ECM-disconnect detection and
late-data handling. Work in progress; findings and reproduction steps follow once the pipeline runs
end to end on the published inputs.
