"""Late data: when offline reports reach the backend, and what a watermark does to trip segmentation.

The export carries event time only. Reports flagged ``is_stored`` were written to the device's
offline buffer and re-sent after it reconnected. The firmware log parser project saw live
reporting resume at reconnection and the stored copies reach the backend 10 to 32 seconds later,
so a stored report's arrival is modelled as the event time of the device's next live report plus
``STORED_REPLAY_DELAY_S`` (20 s, the middle of that range); a stored report with no later live
report in the window arrives after the window ends. Live reports arrive at their event time.
This is a model of arrival, stated as such; it is the only one the export supports.

The delay matters for one strategy. Without it every stored report would arrive before the live
report that follows it, so a per-device consumer would still see its reports in event order and
the ``arrival_order`` strategy would equal the oracle (the first fleet run showed exactly that).
With it the live report arrives first and the backlog after, which is what the backend sees.

Strategies replayed against the oracle (every report, event-time order):

* ``drop_late(L)``: an event-time consumer with a watermark L seconds behind the newest event; a
  report that arrives more than L seconds after its event time is dropped. L = 0 keeps live
  reports only, L = infinity is the oracle.
* ``arrival_order``: a consumer that applies reports in the order they arrive, with no buffering,
  the way a naive pipeline would.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import replay, trips

LATENESS_LEVELS = (0, 30, 60, 300, 900, math.inf)
STORED_REPLAY_DELAY_S = 20.0


def arrival_times(stream: pd.DataFrame, delay_s: float = STORED_REPLAY_DELAY_S) -> pd.DataFrame:
    """Return the stream with ``arrival_ts``, ``lateness_s`` and ``arrival_after_window`` columns."""
    s = stream.sort_values(["device_key", "ts"], kind="mergesort").copy()
    window_end = s["ts"].max()
    delay = pd.Timedelta(seconds=delay_s)
    s["arrival_ts"] = s["ts"]
    s["arrival_after_window"] = False
    stored_idx = []
    arrivals = []
    for _, g in s.groupby("device_key", sort=False):
        st = g.index[g["is_stored"].values]
        if len(st) == 0:
            continue
        live = g.loc[~g["is_stored"], "ts"].dt.tz_convert(None).to_numpy()
        pos = np.searchsorted(live, g.loc[st, "ts"].dt.tz_convert(None).to_numpy(), side="right")
        has = pos < len(live)
        arr = np.where(has, live[np.minimum(pos, max(len(live) - 1, 0))] if len(live) else np.datetime64("NaT", "us"), np.datetime64("NaT", "us"))
        stored_idx.extend(st)
        arrivals.extend(arr)
    if stored_idx:
        arr = pd.to_datetime(pd.Series(arrivals, index=stored_idx), utc=True) + delay
        after = arr.isna()
        arr[after] = window_end + delay
        s.loc[stored_idx, "arrival_ts"] = arr
        s.loc[stored_idx, "arrival_after_window"] = after.values
    s["lateness_s"] = (s["arrival_ts"] - s["ts"]).dt.total_seconds()
    return s.sort_values(["ts", "device_key", "row_id"], kind="mergesort").reset_index(drop=True)


def lateness_summary(s: pd.DataFrame) -> dict:
    """Lateness statistics over the stored reports that arrive inside the window; the ones that do not are counted, not timed."""
    late = s.loc[s["is_stored"] & ~s["arrival_after_window"], "lateness_s"]
    ordered = s.sort_values(["device_key", "ts"], kind="mergesort")
    flag = ordered["is_stored"].astype(int)
    starts = (flag == 1) & (flag.groupby(ordered["device_key"]).shift(fill_value=0) == 0)
    n_runs = int(starts.sum())
    return {
        "reports": int(len(s)), "stored_reports": int(s["is_stored"].sum()), "stored_share": round(float(s["is_stored"].mean()), 4),
        "devices_with_stored": int(s.loc[s["is_stored"], "device_key"].nunique()), "devices": int(s["device_key"].nunique()),
        "offline_episodes": n_runs, "arrive_after_window": int(s["arrival_after_window"].sum()),
        "lateness_s_median": float(late.median()) if len(late) else None, "lateness_s_p90": float(late.quantile(0.9)) if len(late) else None,
        "lateness_s_max": float(late.max()) if len(late) else None,
        "share_within_60s": round(float((late <= 60).mean()), 4) if len(late) else None, "share_within_300s": round(float((late <= 300).mean()), 4) if len(late) else None,
        "share_within_900s": round(float((late <= 900).mean()), 4) if len(late) else None,
    }


def trip_stats(seg: trips.TripSegmenter) -> dict:
    t = seg.trips()
    return {"trips": int(len(t)), "trip_hours": round(float(t["duration_s"].sum() / 3600), 2), "devices_with_trips": int(t["device_key"].nunique()),
            "trips_ended_by_gap": int((t["end_reason"] == "gap").sum()), "stops": int(len(seg.stops()))}


def out_of_order(s: pd.DataFrame) -> int:
    """Reports that reach a per-device consumer with an event time older than one it has already seen, in arrival order."""
    o = replay.ordered(s)
    seen = o.groupby("device_key")["ts"].cummax()
    return int((o["ts"] < seen.groupby(o["device_key"]).shift()).sum())


def _per_device_trip_counts(seg: trips.TripSegmenter) -> pd.Series:
    t = seg.trips()
    return t.groupby("device_key").size() if len(t) else pd.Series(dtype=int)


def compare(oracle: trips.TripSegmenter, other: trips.TripSegmenter) -> dict:
    a, b = _per_device_trip_counts(oracle), _per_device_trip_counts(other)
    idx = a.index.union(b.index)
    a, b = a.reindex(idx, fill_value=0), b.reindex(idx, fill_value=0)
    ta, tb = oracle.trips(), other.trips()
    ha = ta.groupby("device_key")["duration_s"].sum().reindex(idx, fill_value=0.0)
    hb = tb.groupby("device_key")["duration_s"].sum().reindex(idx, fill_value=0.0)
    return {"devices_with_different_trip_count": int((a != b).sum()), "trip_count_abs_diff": int((a - b).abs().sum()), "trip_hours_abs_diff": round(float((ha - hb).abs().sum() / 3600), 2)}


def run_strategies(stream_with_arrival: pd.DataFrame, levels=LATENESS_LEVELS) -> pd.DataFrame:
    """One row per strategy: records kept, trip statistics and the difference to the oracle."""
    s = stream_with_arrival
    oracle = trips.segment(replay.to_reports(s.drop(columns=["arrival_ts"])))
    rows = []
    base = {"strategy": "oracle", "allowed_lateness_s": None, "records_kept": int(len(s)), "records_dropped": 0, "out_of_order": 0, **trip_stats(oracle), **compare(oracle, oracle)}
    rows.append(base)
    for L in levels:
        kept = s[s["lateness_s"] <= L] if math.isfinite(L) else s
        seg = trips.segment(replay.to_reports(kept.drop(columns=["arrival_ts"])))
        rows.append({"strategy": "drop_late", "allowed_lateness_s": None if math.isinf(L) else L, "records_kept": int(len(kept)), "records_dropped": int(len(s) - len(kept)),
                     "out_of_order": 0, **trip_stats(seg), **compare(oracle, seg)})
    seg = trips.segment(replay.to_reports(s))      # arrival order, no reordering
    rows.append({"strategy": "arrival_order", "allowed_lateness_s": None, "records_kept": int(len(s)), "records_dropped": 0, "out_of_order": out_of_order(s),
                 **trip_stats(seg), **compare(oracle, seg)})
    return pd.DataFrame(rows)
