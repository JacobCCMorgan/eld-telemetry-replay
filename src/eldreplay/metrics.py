"""Fleet-level metrics from the consumer observations, the trips and the stream itself."""
from __future__ import annotations

import pandas as pd

FLAGS = ["ecm_silent", "ecm_silent_rpm_or_speed", "err005_shipped", "err005_shipped_sent", "err005_corrected", "err005_corrected_sent", "speed_mismatch", "ecm_frozen",
         "odometer_regress", "odometer_jump", "odometer_zero", "gap_while_driving", "gps_invalid_moving", "sleeping_while_moving"]
STATES = ["driving", "idling", "stationary", "silent", "sleeping", "event"]


def fleet_timeline(obs: pd.DataFrame) -> pd.DataFrame:
    """Devices by derived state per minute (a device counts in the state of its latest report that minute), plus stored and silent shares."""
    o = obs.copy()
    o["minute"] = o["ts"].dt.floor("min")
    last = o.sort_values("ts").groupby(["minute", "device_key"]).tail(1)
    counts = last.pivot_table(index="minute", columns="state", values="device_key", aggfunc="count", fill_value=0)
    for s in STATES:
        if s not in counts:
            counts[s] = 0
    counts = counts[STATES]
    counts["devices_reporting"] = counts.sum(axis=1)
    per_min = o.groupby("minute").agg(reports=("row_id", "size"), stored=("is_stored", "sum"), ecm_silent=("ecm_silent", "sum"), moving=("moving", "sum"))
    counts = counts.join(per_min)
    counts["stored_share"] = (counts["stored"] / counts["reports"]).round(4)
    counts["ecm_silent_share_of_moving"] = (counts["ecm_silent"] / counts["moving"].where(counts["moving"] > 0)).round(4)
    return counts.reset_index()


def flag_summary(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_dev = obs["device_key"].nunique()
    for f in FLAGS:
        hit = obs[obs[f]]
        rows.append({"check": f, "reports": int(len(hit)), "share_of_reports": round(len(hit) / len(obs), 5), "devices": int(hit["device_key"].nunique()),
                     "share_of_devices": round(hit["device_key"].nunique() / n_dev, 4)})
    return pd.DataFrame(rows)


def group_summary(obs: pd.DataFrame, stream: pd.DataFrame) -> pd.DataFrame:
    """Per device batch: devices, reports, driving reports, ECM silence, ERR005 events under both rules, odometer anomalies."""
    m = obs.merge(stream[["row_id", "org_group", "is_internal_test"]], on="row_id")
    g = m.groupby("org_group").agg(devices=("device_key", "nunique"), reports=("row_id", "size"), driving_reports=("moving", "sum"), ecm_silent=("ecm_silent", "sum"),
                                   devices_ecm_silent=("device_key", lambda s: m.loc[s.index][m.loc[s.index, "ecm_silent"]]["device_key"].nunique()),
                                   err005_shipped_events=("err005_shipped_sent", "sum"), err005_corrected_events=("err005_corrected_sent", "sum"),
                                   odometer_regress=("odometer_regress", "sum"), odometer_jump=("odometer_jump", "sum"), odometer_zero=("odometer_zero", "sum"),
                                   speed_mismatch=("speed_mismatch", "sum"), ecm_frozen=("ecm_frozen", "sum"), stored=("is_stored", "sum"))
    g["ecm_silent_share_of_driving"] = (g["ecm_silent"] / g["driving_reports"].where(g["driving_reports"] > 0)).round(4)
    g["stored_share"] = (g["stored"] / g["reports"]).round(4)
    return g.reset_index().sort_values("devices", ascending=False)


def cadence_summary(obs: pd.DataFrame) -> dict:
    gaps = obs["gap_s"].dropna()
    driving_gaps = obs.loc[obs["gap_s"].notna() & obs["state"].isin(["driving", "idling"]), "gap_s"]
    return {"intervals": int(len(gaps)), "median_s": float(gaps.median()), "p90_s": float(gaps.quantile(0.9)), "p99_s": float(gaps.quantile(0.99)),
            "share_within_35s": round(float((gaps <= 35).mean()), 4), "driving_intervals": int(len(driving_gaps)), "driving_share_within_35s": round(float((driving_gaps <= 35).mean()), 4),
            "gaps_over_90s_while_driving": int(obs["gap_while_driving"].sum()), "devices_with_gap_while_driving": int(obs.loc[obs["gap_while_driving"], "device_key"].nunique())}


def unidentified_summary(stream: pd.DataFrame) -> dict:
    ev = stream[stream["event_type"].notna()]
    by = ev["event_type"].value_counts().to_dict()
    ud_dev = ev.loc[ev["event_type"] == "unidentified_driving", "device_key"].nunique()
    no_driver_driving = stream[(stream["mode"] == "Driving") & (stream["driver_key"].isna()) & (stream["gps_speed_kmh"].fillna(0) >= 8)]
    return {"event_records": int(len(ev)), "by_type": {k: int(v) for k, v in by.items()}, "devices_with_unidentified_driving_event": int(ud_dev),
            "moving_reports_without_driver": int(len(no_driver_driving)), "devices_moving_without_driver": int(no_driver_driving["device_key"].nunique()),
            "share_of_moving_reports_without_driver": round(float(len(no_driver_driving) / max(int((stream["gps_speed_kmh"].fillna(0) >= 8).sum()), 1)), 4)}


def trip_summary(trips: pd.DataFrame, stops: pd.DataFrame, stream: pd.DataFrame) -> dict:
    t = trips
    full = t[~t["truncated_start"] & ~t["truncated_end"]]
    ordered = stream.sort_values(["device_key", "ts"])
    transitions = int(((ordered.groupby("device_key")["mode"].shift() == "Driving") & (ordered["mode"] == "Sleeping")).sum())
    return {"trips": int(len(t)), "devices_with_trips": int(t["device_key"].nunique()), "complete_trips": int(len(full)),
            "truncated_start": int(t["truncated_start"].sum()), "truncated_end": int(t["truncated_end"].sum()),
            "end_reasons": {k: int(v) for k, v in t["end_reason"].value_counts().items()}, "trip_hours": round(float(t["duration_s"].sum() / 3600), 1),
            "complete_trip_minutes_median": round(float(full["duration_s"].median() / 60), 1) if len(full) else None,
            "complete_trip_miles_median": round(float(full["distance_miles"].median()), 1) if full["distance_miles"].notna().any() else None,
            "stops": int(len(stops)), "stop_minutes_median": round(float(stops["duration_s"].median() / 60), 1) if len(stops) else None,
            "driving_to_sleeping_transitions": transitions}


def tier_check(device_summary: pd.DataFrame, priority: pd.DataFrame, stream: pd.DataFrame) -> pd.DataFrame:
    """The ECM-disconnect project's tiers against this stream: devices seen, driving reports, ECM-silent share."""
    d = device_summary.set_index("device_key")
    p = priority.set_index("device_key")
    p["in_stream"] = p.index.isin(stream["device_key"])
    for col, src in (("reports", "reports"), ("driving_reports", "state_driving"), ("ecm_silent", "ecm_silent")):
        p[col] = d[src].reindex(p.index).fillna(0) if src in d else 0
    g = p.groupby("tier").agg(devices=("in_stream", "size"), in_stream=("in_stream", "sum"), reports=("reports", "sum"), driving_reports=("driving_reports", "sum"), ecm_silent=("ecm_silent", "sum"),
                              devices_silent=("ecm_silent", lambda s: int((s > 0).sum())))
    g["ecm_silent_share_of_driving"] = (g["ecm_silent"] / g["driving_reports"].where(g["driving_reports"] > 0)).round(3)
    return g.reset_index()
