"""Overnight aircraft positioning features (EXP-003, backlog idea F2).

For each RDU flight on date D:
Finds the aircraft's final leg on date D-1 across the national US schedule.
Point-in-time rule (HANDOFF §2.1):
Actual arrival outcomes from day D-1 are only known if outcome_time_utc < 23:00 ET cutoff.
Flights still airborne at 23:00 ET have scheduled information only.

Features:
- prev_day_tail_seen: 1 if tail flew on day D-1, 0 otherwise
- prev_day_dest_is_rdu: 1 if yesterday's last scheduled destination was RDU, 0 if elsewhere
- prev_day_completed_before_cutoff: 1 if yesterday's last leg finished before 23:00 ET
- prev_day_last_arr_delay: actual arrival delay (min, capped 180) if completed before cutoff
- prev_day_ended_late15: 1 if completed before cutoff with arr_delay >= 15 min
- prev_day_ended_late45: 1 if completed before cutoff with arr_delay >= 45 min
- prev_day_overnight_sched_turn_min: scheduled turn minutes between yesterday's last arrival and today's departure
- prev_day_n_legs: number of legs the airframe flew on day D-1
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rdu.eval.walkforward import ROOT, Data, cutoff_utc

STORE = ROOT / "data" / "features" / "position.parquet"


def build(data: Data) -> pd.DataFrame:
    flights = data.flights
    legs = data.legs[data.legs["tail"].notna()].sort_values(["flight_date", "tail", "sched_dep_utc"])

    # Aggregate yesterday's legs per airframe
    g = legs.groupby(["flight_date", "tail"])
    last_legs = g.last().reset_index()
    last_legs["n_legs_yest"] = g.size().to_numpy()
    last_legs["next_date"] = last_legs["flight_date"] + timedelta(days=1)

    # Merge with today's RDU flights
    merged = pd.merge(
        flights[["flight_id", "forecast_date", "tail", "sched_dep_utc", "direction"]],
        last_legs[["next_date", "tail", "dest", "arr_delay", "outcome_time_utc", "sched_arr_utc", "n_legs_yest"]],
        left_on=["forecast_date", "tail"],
        right_on=["next_date", "tail"],
        how="left",
    )
    merged.index = flights.index

    # Precompute cutoff times for each unique forecast date (point-in-time rule)
    u_dates = pd.Series(flights["forecast_date"].unique())
    dmap = dict(zip(u_dates, u_dates.map(cutoff_utc)))
    cutoffs = flights["forecast_date"].map(dmap)

    tail_seen = merged["next_date"].notna()
    dest_is_rdu = (merged["dest"] == "RDU").astype(float).where(tail_seen)
    
    # Finished before 23:00 ET cutoff on D-1
    finished = (merged["outcome_time_utc"] < cutoffs).astype(float).where(tail_seen)
    
    # Delay metrics only available if completed before cutoff
    delay_known = (finished == 1.0) & merged["arr_delay"].notna()
    arr_delay = np.clip(merged["arr_delay"], -60, 180).where(delay_known)
    late15 = (merged["arr_delay"] >= 15.0).astype(float).where(delay_known)
    late45 = (merged["arr_delay"] >= 45.0).astype(float).where(delay_known)

    # Scheduled overnight turn minutes between yesterday's last arrival and today's first departure
    t_diff = (merged["sched_dep_utc"] - merged["sched_arr_utc"]).dt.total_seconds() / 60.0
    overnight_turn = np.clip(t_diff, 0, 1440).where(tail_seen)

    out = pd.DataFrame({
        "prev_day_tail_seen": tail_seen.astype(float),
        "prev_day_dest_is_rdu": dest_is_rdu,
        "prev_day_completed_before_cutoff": finished,
        "prev_day_last_arr_delay": arr_delay,
        "prev_day_ended_late15": late15,
        "prev_day_ended_late45": late45,
        "prev_day_overnight_turn_min": overnight_turn,
        "prev_day_n_legs": merged["n_legs_yest"].astype(float).where(tail_seen),
    })
    out.index = flights["flight_id"].to_numpy()

    return out.astype("float32")



def load() -> pd.DataFrame:
    return pd.read_parquet(STORE)


if __name__ == "__main__":
    from rdu.eval.walkforward import load_data
    data = load_data()
    X = build(data)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(STORE)
    print(f"Built position features: {STORE} {X.shape}")
