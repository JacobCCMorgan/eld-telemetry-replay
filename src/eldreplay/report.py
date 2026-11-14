"""Run the whole pipeline on a stream and write the report, figures and tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import consumer, latedata, metrics, replay, trips  # noqa: E402

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3de"
STATE_COLORS = {"driving": "#2a78d6", "idling": "#eb6834", "stationary": "#eda100", "silent": "#e34948", "sleeping": "#b8b7b0", "event": "#6250d6"}
TIER_COLORS = {"P1": "#e34948", "P2": "#eb6834", "P3": "#eda100", "P4": "#2a78d6"}
LOCAL_TZ = "America/Los_Angeles"


def _style(ax, xgrid=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7b0")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def analyze(stream: pd.DataFrame, priority: pd.DataFrame | None = None, lateness_levels=latedata.LATENESS_LEVELS) -> dict:
    """Everything the report needs, as frames and dicts."""
    cons = consumer.consume(replay.to_reports(stream))
    obs = cons.observations()
    dev = cons.device_summary()
    seg = trips.segment(replay.to_reports(stream))
    t, st = seg.trips(), seg.stops()
    with_arrival = latedata.arrival_times(stream)
    stored_cols = ["device_key", "ts", "arrival_ts", "lateness_s", "arrival_after_window"]
    out = {
        "observations": obs, "device_summary": dev, "trips": t, "stops": st,
        "timeline": metrics.fleet_timeline(obs), "flags": metrics.flag_summary(obs), "groups": metrics.group_summary(obs, stream),
        "cadence": metrics.cadence_summary(obs), "unidentified": metrics.unidentified_summary(stream), "trip_summary": metrics.trip_summary(t, st, stream),
        "lateness": latedata.lateness_summary(with_arrival), "strategies": latedata.run_strategies(with_arrival, lateness_levels),
        "stored_lateness": with_arrival.loc[with_arrival["is_stored"], stored_cols].reset_index(drop=True),
    }
    if priority is not None:
        out["tiers"] = metrics.tier_check(dev, priority, stream)
    return out


def fig_timeline(a: dict, path: Path):
    tl = a["timeline"].set_index("minute")
    tl = tl[tl["devices_reporting"] >= 0.5 * tl["devices_reporting"].median()]      # drop the partial first or last minute of the window
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=SURFACE)
    _style(ax)
    x = tl.index.tz_convert(LOCAL_TZ)
    keys = ["driving", "idling", "stationary", "silent", "sleeping", "event"]
    ax.stackplot(x, [tl[k] for k in keys], labels=keys, colors=[STATE_COLORS[k] for k in keys], linewidth=0)
    ax.set_ylabel("devices (state of the latest report in the minute)", color=INK2)
    ax.set_title("Fleet by state, per minute (Pacific time, UTC-7)", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, frameon=False, fontsize=9)
    ax.margins(x=0)
    fig.autofmt_xdate()
    _save(fig, path)


def fig_positions(stream: pd.DataFrame, path: Path):
    pos = stream.dropna(subset=["lat_coarse", "lon_coarse"])
    pos = pos[(pos["lat_coarse"].between(18, 72)) & (pos["lon_coarse"].between(-170, -50))]
    cells = pos.groupby(["lon_coarse", "lat_coarse"])["device_key"].nunique().reset_index(name="devices")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
    _style(ax, xgrid=True)
    ax.scatter(cells["lon_coarse"], cells["lat_coarse"], s=6 + 4 * np.sqrt(cells["devices"]), c="#2a78d6", alpha=0.55, linewidths=0)
    ax.set_xlabel("longitude (0.05-degree grid)", color=INK2)
    ax.set_ylabel("latitude", color=INK2)
    title = f"Where the fleet was: {len(cells):,} grid cells, {pos['device_key'].nunique():,} devices with a fix (marker area = devices in the cell)"
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_aspect(1.3)
    _save(fig, path)


def fig_ecm_by_group(a: dict, path: Path):
    g = a["groups"].dropna(subset=["ecm_silent_share_of_driving"]).sort_values("ecm_silent_share_of_driving")
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    _style(ax, xgrid=True)
    ax.yaxis.grid(False)
    y = list(range(len(g)))
    ax.barh(y, g["ecm_silent_share_of_driving"] * 100, color="#2a78d6", height=0.62, linewidth=0)
    for i, row in enumerate(g.itertuples()):
        share = row.ecm_silent_share_of_driving * 100
        ax.text(share + 0.4, i, f"{share:.1f} %  ({row.devices_ecm_silent} of {row.devices} devices)", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, g["org_group"])
    ax.set_xlim(0, max(g["ecm_silent_share_of_driving"].max() * 100 + 12, 10))
    ax.set_xlabel("share of moving reports with no RPM from the ECM", color=INK2)
    ax.set_title("ECM silence while moving, by device batch", loc="left", fontsize=11, color=INK)
    _save(fig, path)


def fig_tiers(a: dict, path: Path):
    if "tiers" not in a:
        return
    t = a["tiers"]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _style(ax)
    x = list(range(len(t)))
    ax.bar(x, t["ecm_silent_share_of_driving"].fillna(0) * 100, color=[TIER_COLORS.get(k, "#2a78d6") for k in t["tier"]], width=0.62, linewidth=0)
    for i, row in enumerate(t.itertuples()):
        share = (row.ecm_silent_share_of_driving or 0) * 100
        ax.text(i, share + 1.5, f"{share:.0f} %\n{int(row.in_stream)} of {int(row.devices)} in stream", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"tier {k}" for k in t["tier"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("ECM-silent share of driving reports", color=INK2)
    ax.set_title("The ECM-disconnect project's tiers, measured on this stream (an independent data path)", loc="left", fontsize=11, color=INK)
    _save(fig, path)


def fig_lateness(a: dict, path: Path):
    late = a["stored_lateness"]
    late = late.loc[~late["arrival_after_window"], "lateness_s"] / 60
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    _style(ax)
    ax.hist(late.clip(upper=29), bins=np.arange(0, 30, 1), color="#2a78d6", linewidth=0)
    ax.set_xlabel("minutes between event time and modelled arrival (29 = 29 or more)", color=INK2)
    ax.set_ylabel("stored reports", color=INK2)
    ax.set_title(f"How late offline reports arrive ({len(late):,} stored reports; median {late.median():.1f} min)", loc="left", fontsize=11, color=INK)
    ax = axes[1]
    _style(ax)
    s = a["strategies"]
    d = s[s["strategy"] == "drop_late"].copy()
    d["label"] = [("none" if pd.isna(v) else f"{int(v)} s") + f"\n{int(n):,} dropped" for v, n in zip(d["allowed_lateness_s"], d["records_dropped"], strict=True)]
    arr = s[s["strategy"] == "arrival_order"]
    if len(arr):
        arr = arr.assign(label=[f"unbuffered\n{int(arr['out_of_order'].iloc[0]):,} reordered"])
        d = pd.concat([d, arr], ignore_index=True)
    x = list(range(len(d)))
    colors = ["#6250d6" if st == "arrival_order" else "#eb6834" for st in d["strategy"]]
    ax.bar(x, d["devices_with_different_trip_count"], color=colors, width=0.62, linewidth=0)
    for i, row in enumerate(d.itertuples()):
        ax.text(i, row.devices_with_different_trip_count + 1, f"{row.devices_with_different_trip_count}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, d["label"], fontsize=8)
    ax.set_xlabel("allowed lateness (watermark), or no buffering at all", color=INK2)
    ax.set_ylabel("devices whose trip count differs from the oracle", color=INK2)
    ax.set_title("Dropping late reports, or applying them unbuffered, versus the oracle", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, max(d["devices_with_different_trip_count"].max() * 1.25, 5))
    _save(fig, path)


def fig_trips(a: dict, path: Path):
    t = a["trips"]
    full = t[~t["truncated_start"] & ~t["truncated_end"]]
    st = a["stops"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    ax.hist((full["duration_s"] / 60).clip(upper=60), bins=np.arange(0, 62, 2), color="#2a78d6", linewidth=0)
    ax.set_xlabel("complete trip duration, minutes (60 = 60 or more)", color=INK2)
    ax.set_ylabel("trips", color=INK2)
    ax.set_title(f"{len(full):,} complete trips inside the window (median {full['duration_s'].median() / 60:.1f} min)", loc="left", fontsize=11, color=INK)
    ax = axes[1]
    _style(ax)
    ax.hist((st["duration_s"] / 60).clip(upper=120), bins=np.arange(0, 124, 4), color="#eb6834", linewidth=0)
    ax.set_xlabel("stop duration, minutes (120 = 120 or more)", color=INK2)
    ax.set_ylabel("stops", color=INK2)
    ax.set_title(f"{len(st):,} stops of five minutes or more (median {st['duration_s'].median() / 60:.1f} min)", loc="left", fontsize=11, color=INK)
    _save(fig, path)


def _cell(v, floatfmt: str) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{int(v):,}" if float(v).is_integer() and abs(v) >= 1000 else floatfmt.format(v)
    return str(v)


def _md_table(df: pd.DataFrame, cols=None, floatfmt="{:.3f}") -> str:
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(r[c], floatfmt) for c in cols) + " |")
    return "\n".join(lines)


def _pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f} %"


def _findings(stream: pd.DataFrame, a: dict, summary: dict) -> list[str]:
    s, c, u, t, L = summary["stream"], a["cadence"], a["unidentified"], a["trip_summary"], a["lateness"]
    g = a["groups"].set_index("org_group")
    fl = a["flags"].set_index("check")
    tl = a["timeline"]
    strat = a["strategies"]
    oracle = strat[strat["strategy"] == "oracle"].iloc[0]
    drops = strat[strat["strategy"] == "drop_late"]
    drop0 = drops[drops["allowed_lateness_s"] == 0].iloc[0]
    drop300 = drops[drops["allowed_lateness_s"] == 300].iloc[0] if (drops["allowed_lateness_s"] == 300).any() else drop0
    arrival = strat[strat["strategy"] == "arrival_order"].iloc[0]
    tiers = a.get("tiers")
    gshare = g["ecm_silent_share_of_driving"]
    biggest = g["devices"].idxmax()
    obs = a["observations"]
    frozen_dev = obs.loc[obs["ecm_frozen"], "device_key"].value_counts()
    worst = frozen_dev.index[0] if len(frozen_dev) else None
    worst_mismatch = int(obs.loc[(obs["device_key"] == worst) & obs["speed_mismatch"]].shape[0]) if worst else 0
    worst_group = stream.loc[stream["device_key"] == worst, "org_group"].iloc[0] if worst else "-"
    frozen_devices = int(fl.loc["ecm_frozen", "devices"])
    ev = u["by_type"]
    out = []
    out.append(" ".join([
        f"**The fleet at a glance.** In a typical minute {int(tl['devices_reporting'].median()):,} devices report: about {int(tl['driving'].median()):,} moving,",
        f"{int(tl['idling'].median()):,} with the engine running but not moving, {int(tl['sleeping'].median()):,} asleep.",
        f"Driving reports keep the 30-second cadence: {_pct(c['driving_share_within_35s'], 1)} of intervals are 35 s or shorter and only",
        f"{c['gaps_over_90s_while_driving']} gaps over 90 s occurred, on {c['devices_with_gap_while_driving']} devices.",
    ]))
    out.append(" ".join([
        f"**ECM silence while moving is rare fleet-wide and concentrated in batches.** {fl.loc['ecm_silent', 'reports']:,} reports",
        f"({_pct(fl.loc['ecm_silent', 'reports'] / s['reports'], 2)}) on {fl.loc['ecm_silent', 'devices']} devices ({_pct(fl.loc['ecm_silent', 'devices'] / s['devices'], 1)})",
        "carried GPS motion with no RPM. The share of moving reports affected is",
        f"{_pct(gshare.get('Test Group', 0))} in the provider's test group, {_pct(gshare.get('Volvo 2026', 0))} in the Volvo 2026 batch and",
        f"{_pct(gshare.get('OBD II', 0))} on OBD-II units, against {_pct(gshare[biggest], 1)} in the largest customer batch ({biggest}).",
    ]))
    if tiers is not None:
        tt = tiers.set_index("tier").reindex(["P1", "P2", "P3", "P4"]).fillna(0)
        share = tt["ecm_silent_share_of_driving"]
        out.append(" ".join([
            "**The ECM-disconnect project's ranking holds on an independent data path.** Its tiers were built from the platform's anomaly export;",
            f"measured on the raw device stream, tier P1 devices are ECM-silent in {_pct(share['P1'])} of their driving reports,",
            f"P2 in {_pct(share['P2'])}, P3 in {_pct(share['P3'])} and P4 in {_pct(share['P4'], 1)}",
            f"({int(tt.loc['P1', 'in_stream'])} of {int(tt.loc['P1', 'devices'])} P1 devices reported in the window).",
        ]))
    out.append(" ".join([
        "**The corrected ERR005 rule changes almost nothing at fleet level.** With the firmware's one-hour cooldown, the shipped rule (any ECM field missing) sends",
        f"{fl.loc['err005_shipped_sent', 'reports']} events from {fl.loc['err005_shipped_sent', 'devices']} devices in two hours; the speed-and-RPM rule sends",
        f"{fl.loc['err005_corrected_sent', 'reports']} from {fl.loc['err005_corrected_sent', 'devices']}. The OBD-II false positive found on one test vehicle in the",
        "firmware log parser project (engine hours never available) is not the fleet's pattern: here engine hours are missing almost only when speed and RPM are missing too.",
    ]))
    j1708 = int(g.loc["1708", "odometer_regress"] + g.loc["1708", "odometer_jump"]) if "1708" in g.index else 0
    out.append(" ".join([
        f"**Odometer anomalies are a protocol problem.** {fl.loc['odometer_regress', 'reports']} regressions and {fl.loc['odometer_jump', 'reports']} impossible jumps",
        f"between valid readings, {j1708} of them on the {int(g.loc['1708', 'devices']) if '1708' in g.index else 0} J1708 units; the J1939 batches contribute single digits.",
        f"Separately, {fl.loc['odometer_zero', 'reports']} driving or idling reports carry an odometer of 0 (the device writes 0 when the ECM does not answer),",
        f"{int(g['odometer_zero'].get('Test Group', 0))} of them in the test group and {int(g['odometer_zero'].get('RT67', 0))} in batch RT67.",
    ]))
    out.append(" ".join([
        f"**One device drove for two hours without moving.** GPS and wheel speed disagree by more than 15 km/h in {fl.loc['speed_mismatch', 'reports']} reports on",
        f"{fl.loc['speed_mismatch', 'devices']} devices; {worst_mismatch} of them are one device (batch {worst_group}) whose ECM frame is frozen: the same speed and RPM",
        f"in every report of the window while a valid GPS fix shows it standing still. The `ecm_frozen` check raises {fl.loc['ecm_frozen', 'reports']} reports on",
        f"{frozen_devices} device{'s' if frozen_devices != 1 else ''} in total; every one of those reports counts as driving in the platform's mileage and",
        "hours-of-service totals. The other mismatches are mostly single reports around acceleration and braking, where the two sources lag each other.",
    ]))
    out.append(" ".join([
        f"**Half of the moving fleet has no driver logged in.** {u['moving_reports_without_driver']:,} moving reports ({_pct(u['share_of_moving_reports_without_driver'])}) from",
        f"{u['devices_moving_without_driver']:,} devices carry no driver id, and {u['devices_with_unidentified_driving_event']} devices raised an unidentified-driving event",
        f"within the two hours ({ev.get('unidentified_driving', 0)} start, {ev.get('unidentified_driving_diagnose', 0)} diagnostic,",
        f"{ev.get('unidentified_driving_clear_speed', 0) + ev.get('unidentified_driving_clear_driver', 0)} cleared).",
        "The logbook toolkit project saw the same thing on one driver's file; this is the fleet-wide rate.",
    ]))
    out.append(" ".join([
        f"**Trips and stops.** {t['trips']:,} trips on {t['devices_with_trips']:,} devices, {t['complete_trips']:,} of them complete inside the window",
        f"(median {t['complete_trip_minutes_median']} min, {t['complete_trip_miles_median']} miles: short moves dominate a two-hour slice),",
        f"{t['stops']:,} stops of five minutes or more (median {t['stop_minutes_median']} min). The segmenter reproduces the",
        f"{t['driving_to_sleeping_transitions']:,} Driving-to-Sleeping transitions the devices themselves reported.",
    ]))
    out.append(" ".join([
        f"**Late data is small but not negligible, and ordering matters more than lateness.** {L['stored_reports']:,} reports ({_pct(L['stored_share'], 1)}) were written to the",
        f"offline buffer, on {L['devices_with_stored']:,} devices in {L['offline_episodes']:,} episodes; under the reconnection model they arrive a median",
        f"{L['lateness_s_median'] / 60:.1f} min late, {_pct(L['share_within_300s'])} within five minutes, the slowest after {L['lateness_s_max'] / 60:.0f} min",
        f"({L['arrive_after_window']} arrive after the window). A consumer that drops everything late (watermark 0) loses {int(drop0['records_dropped']):,} reports and",
        f"changes the trip count of {int(drop0['devices_with_different_trip_count'])} devices; a five-minute watermark leaves {int(drop300['devices_with_different_trip_count'])}",
        f"devices different. A consumer that applies reports in arrival order without buffering keeps every report but sees {int(arrival['out_of_order']):,} of them",
        f"out of order and still gets {int(arrival['devices_with_different_trip_count'])} devices wrong ({int(arrival['trips_ended_by_gap'])} trips cut by phantom gaps,",
        f"against {int(oracle['trips_ended_by_gap'])} in the oracle).",
    ]))
    return [f"{i}. {text}" for i, text in enumerate(out, 1)]


def write(stream: pd.DataFrame, a: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    fig_timeline(a, out / "figures" / "01_fleet_timeline.png")
    fig_positions(stream, out / "figures" / "02_positions.png")
    fig_ecm_by_group(a, out / "figures" / "03_ecm_silence_by_batch.png")
    fig_tiers(a, out / "figures" / "04_priority_tiers_on_stream.png")
    fig_lateness(a, out / "figures" / "05_late_data.png")
    fig_trips(a, out / "figures" / "06_trips_and_stops.png")
    a["timeline"].to_csv(out / "tables" / "fleet_timeline.csv", index=False)
    a["flags"].to_csv(out / "tables" / "checks_summary.csv", index=False)
    a["groups"].to_csv(out / "tables" / "batch_summary.csv", index=False)
    a["device_summary"].to_csv(out / "tables" / "device_summary.csv", index=False)
    a["trips"].to_parquet(out / "tables" / "trips.parquet", index=False)
    a["stops"].to_parquet(out / "tables" / "stops.parquet", index=False)
    a["strategies"].to_csv(out / "tables" / "late_data_strategies.csv", index=False)
    if "tiers" in a:
        a["tiers"].to_csv(out / "tables" / "priority_tiers_on_stream.csv", index=False)
    summary = {
        "stream": {"reports": int(len(stream)), "devices": int(stream["device_key"].nunique()),
                   "window_start_utc": stream["ts"].min().isoformat(), "window_end_utc": stream["ts"].max().isoformat()},
        "cadence": a["cadence"], "unidentified": a["unidentified"], "trips": a["trip_summary"], "late_data": a["lateness"],
        "checks": {r["check"]: {"reports": int(r["reports"]), "devices": int(r["devices"])} for _, r in a["flags"].iterrows()},
    }
    (out / "tables" / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    s = summary["stream"]
    md = ["# Replaying two hours of fleet telemetry: what a streaming consumer sees", ""]
    md.append(" ".join([
        f"Stream: {s['reports']:,} reports from {s['devices']:,} devices, {s['window_start_utc']} to {s['window_end_utc']} (UTC), replayed in event-time order",
        "through the per-device consumer, the trip segmenter and the late-data strategies. Figures and tables are regenerated by `eldreplay report`.",
    ]))
    md += ["", "## Findings", ""]
    md += _findings(stream, a, summary)
    md += ["", "![fleet timeline](figures/01_fleet_timeline.png)", "", "![positions](figures/02_positions.png)", "",
           "![ECM silence by batch](figures/03_ecm_silence_by_batch.png)", ""]
    if "tiers" in a:
        md += ["![priority tiers on the stream](figures/04_priority_tiers_on_stream.png)", ""]
    md += ["![late data](figures/05_late_data.png)", "", "![trips and stops](figures/06_trips_and_stops.png)", ""]

    md += ["## 1. Consumer checks", "", _md_table(a["flags"], ["check", "reports", "share_of_reports", "devices", "share_of_devices"], "{:.4f}"), ""]
    group_cols = ["org_group", "devices", "reports", "driving_reports", "ecm_silent", "devices_ecm_silent", "ecm_silent_share_of_driving", "err005_shipped_events",
                  "err005_corrected_events", "odometer_regress", "odometer_jump", "odometer_zero", "speed_mismatch", "ecm_frozen", "stored_share"]
    md += ["## 2. By device batch", "", _md_table(a["groups"], group_cols, "{:.4f}"), ""]
    if "tiers" in a:
        tier_cols = ["tier", "devices", "in_stream", "reports", "driving_reports", "ecm_silent", "devices_silent", "ecm_silent_share_of_driving"]
        md += ["## 3. The ECM-disconnect tiers on this stream", "", _md_table(a["tiers"], tier_cols), ""]
    md += ["## 4. Cadence, unidentified driving, trips", "", "```", json.dumps({"cadence": a["cadence"], "unidentified": a["unidentified"], "trips": a["trip_summary"]}, indent=2), "```", ""]
    strat_cols = ["strategy", "allowed_lateness_s", "records_kept", "records_dropped", "out_of_order", "trips", "trip_hours", "trips_ended_by_gap",
                  "devices_with_different_trip_count", "trip_count_abs_diff", "trip_hours_abs_diff"]
    md += ["## 5. Late data", "", "```", json.dumps(a["lateness"], indent=2), "```", "", _md_table(a["strategies"], strat_cols, "{:.0f}"), ""]
    md += ["## Method notes", ""]
    md += [
        "* Replay: reports ordered by (arrival, event time, device, row); arrival = event time for live reports, the device's next live report plus a 20 s replay delay for stored ones",
        "  (a model, see `eldreplay.latedata`).",
        "* Derived state: driving when GPS or wheel speed is at least 8 km/h (the firmware's trigger), idling when the engine reports but the vehicle does not move, silent when",
        "  neither speed nor RPM is reported, sleeping and event records as reported by the device.",
        "* ECM silence: GPS speed above 0 with no RPM, the definition of the ECM-disconnect project. ERR005 shipped rule: GPS speed above 0 and any of speed, RPM, engine hours,",
        "  engine miles missing; corrected rule: speed or RPM missing; both with the firmware's one-hour per-device cooldown.",
        "* Odometer: zero readings are skipped; a regression is a decrease between valid readings, a jump an increase faster than 300 km/h.",
        "* Frozen ECM frame: wheel speed at or above 8 km/h, a valid GPS fix below 1 km/h, and speed and RPM identical to the previous report.",
        "* Trips: start at the first moving report, end at a five-minute stationary span, a Sleeping report or a ten-minute silence; stops are stationary spans of five minutes",
        "  or more between trips.",
        "* Coordinates are on a 0.05-degree grid, so distances come from the odometer, not from positions.",
        "",
    ]
    path = out / "replay_report.md"
    path.write_text("\n".join(md), encoding="ascii")
    return path
