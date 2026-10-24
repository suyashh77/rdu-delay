"""Clean BTS chain files into one legs table and one RDU-flights table.

Times: BTS gives local clock times. Scheduled departure is localized at the
origin's timezone and converted to UTC. Scheduled arrival = scheduled departure
+ CRSElapsedTime, and actual times = scheduled + delay. That sidesteps midnight
crossings entirely, because BTS delays are already relative to the schedule.
"""
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

ET = "America/New_York"
CAP = 180  # minutes; cap for the delay distribution (HANDOFF 2.3, M17)
BUCKETS = ["on_time", "d15_44", "d45_89", "d90_plus", "cancelled"]
# Columns that are only known after the flight happens. Everything else is schedule.
OUTCOME_COLS = [
    "dep_delay", "arr_delay", "cancelled", "diverted", "cancellation_code",
    "actual_dep_utc", "actual_arr_utc", "outcome_time_utc",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
]
# Diverted flights have no arrival time; treat the outcome as known this long after schedule.
DIVERTED_KNOWN_AFTER = pd.Timedelta(hours=6)
# Codes newer than the airportsdata table. DJT = Palm Beach Intl (BTS city name), seen from 2026-08.
TZ_OVERRIDES = {"DJT": "America/New_York"}


def _local_to_utc(date: pd.Series, hhmm: pd.Series, airport: pd.Series) -> pd.Series:
    """Local wall-clock (date + hhmm) at each airport -> UTC timestamp."""
    tz_of = {**{k: v["tz"] for k, v in airportsdata.load("IATA").items()}, **TZ_OVERRIDES}
    tz = airport.map(tz_of)
    missing = sorted(airport[tz.isna()].unique())
    if missing:
        raise ValueError(f"no timezone for airports: {missing}")
    hhmm = hhmm.astype(int)
    naive = (pd.to_datetime(date) + pd.to_timedelta(hhmm // 100, unit="h")
             + pd.to_timedelta(hhmm % 100, unit="m"))
    out = pd.Series(pd.NaT, index=date.index, dtype="datetime64[ns, UTC]")
    for zone, idx in tz.groupby(tz).groups.items():
        out.loc[idx] = (naive.loc[idx]
                        .dt.tz_localize(zone, ambiguous=False, nonexistent="shift_forward")
                        .dt.tz_convert("UTC"))
    return out


def load_raw(bts_dir: Path) -> pd.DataFrame:
    files = sorted(Path(bts_dir).glob("bts_rdu_chains_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no BTS parquet files in {bts_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def clean_legs(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per leg, UTC times, plus aircraft-day chain fields."""
    d = pd.DataFrame({
        "flight_date": pd.to_datetime(raw["FlightDate"]).dt.date,
        "carrier": raw["Reporting_Airline"].astype(str),
        "flight_number": raw["Flight_Number_Reporting_Airline"].astype("Int64"),  # 1 leg (F9 MIA-ATL 2024-08-25) has none
        "tail": raw["Tail_Number"].astype("string"),
        "origin": raw["Origin"].astype(str),
        "dest": raw["Dest"].astype(str),
        "crs_dep_hhmm": raw["CRSDepTime"].astype(int),
        "crs_arr_hhmm": raw["CRSArrTime"].astype(int),
        "crs_elapsed_min": raw["CRSElapsedTime"],
        "distance": raw["Distance"],
        "dep_delay": raw["DepDelay"],
        "arr_delay": raw["ArrDelay"],
        "cancelled": raw["Cancelled"].fillna(0).astype(bool),
        "diverted": raw["Diverted"].fillna(0).astype(bool),
        "cancellation_code": raw["CancellationCode"],
        "carrier_delay": raw["CarrierDelay"],
        "weather_delay": raw["WeatherDelay"],
        "nas_delay": raw["NASDelay"],
        "security_delay": raw["SecurityDelay"],
        "late_aircraft_delay": raw["LateAircraftDelay"],
        "touches_rdu": raw["touches_rdu"].astype(bool),
    })
    n0 = len(d)
    d = d.drop_duplicates(subset=["flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_hhmm"])
    if len(d) != n0:
        print(f"dropped {n0 - len(d)} exact duplicate legs")

    d["sched_dep_utc"] = _local_to_utc(pd.Series(d["flight_date"]), d["crs_dep_hhmm"], d["origin"])
    elapsed = pd.to_timedelta(d["crs_elapsed_min"], unit="m")
    # Rare missing CRSElapsedTime: fall back to local arrival clock, rolled forward past departure.
    fallback = _local_to_utc(pd.Series(d["flight_date"]), d["crs_arr_hhmm"], d["dest"])
    fallback = fallback.where(fallback >= d["sched_dep_utc"], fallback + pd.Timedelta(days=1))
    d["sched_arr_utc"] = (d["sched_dep_utc"] + elapsed).fillna(fallback)
    d["actual_dep_utc"] = d["sched_dep_utc"] + pd.to_timedelta(d["dep_delay"], unit="m")
    d["actual_arr_utc"] = d["sched_arr_utc"] + pd.to_timedelta(d["arr_delay"], unit="m")
    d["outcome_time_utc"] = d["actual_arr_utc"]
    d.loc[d["cancelled"], "outcome_time_utc"] = d.loc[d["cancelled"], "sched_dep_utc"]
    div = d["diverted"] & d["actual_arr_utc"].isna()
    d.loc[div, "outcome_time_utc"] = d.loc[div, "sched_arr_utc"] + DIVERTED_KNOWN_AFTER
    bad = d["outcome_time_utc"].isna()
    if bad.any():
        # Neither arrived, cancelled nor diverted: outcome unknowable; keep row, never treat as known.
        print(f"{int(bad.sum())} legs have no outcome time (kept, never known)")
    return add_chains(d)


def add_chains(d: pd.DataFrame) -> pd.DataFrame:
    """Leg number within each aircraft-day (BTS flight_date x tail), ordered by scheduled departure."""
    d = d.sort_values(["flight_date", "tail", "sched_dep_utc"]).reset_index(drop=True)
    has_tail = d["tail"].notna()
    g = d[has_tail].groupby(["flight_date", "tail"], sort=False)
    d["leg_number"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    d.loc[has_tail, "leg_number"] = g.cumcount() + 1
    d.loc[has_tail, "n_legs_day"] = g["origin"].transform("size")
    d["prev_dest"] = g["dest"].shift(1)
    d["prev_origin"] = g["origin"].shift(1)
    d["prev_sched_arr_utc"] = g["sched_arr_utc"].shift(1)
    d["sched_turn_min"] = (d["sched_dep_utc"] - d["prev_sched_arr_utc"]).dt.total_seconds() / 60
    # A break means the previous leg did not end where this one starts (data gap or swap).
    d["chain_break"] = d["prev_dest"].notna() & (d["prev_dest"] != d["origin"])
    d["overlap"] = d["sched_turn_min"] < 0
    return d


def rdu_flights(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per RDU departure and per RDU arrival, with forecast date (RDU local) and target."""
    rows = []
    for direction, mask, delay_col, time_col in [
        ("dep", legs["origin"] == "RDU", "dep_delay", "sched_dep_utc"),
        ("arr", legs["dest"] == "RDU", "arr_delay", "sched_arr_utc"),
    ]:
        r = legs[mask].copy()
        r["direction"] = direction
        r["leg_index"] = r.index  # row id into legs
        r["sched_rdu_utc"] = r[time_col]
        local = r[time_col].dt.tz_convert(ET)
        r["forecast_date"] = local.dt.date
        r["rdu_local_hour"] = local.dt.hour
        r["delay_min"] = r[delay_col]
        rows.append(r)
    f = pd.concat(rows, ignore_index=True)
    f["bucket"] = bucket_of(f["delay_min"], f["cancelled"], f["diverted"])
    f["y_capped"] = capped_target(f["delay_min"], f["cancelled"], f["diverted"])
    return f


def bucket_of(delay: pd.Series, cancelled: pd.Series, diverted: pd.Series) -> pd.Series:
    """Index into BUCKETS. Diverted with no delay value counts as 90+ (DL-016)."""
    d = delay.to_numpy(dtype=float)
    b = np.select([d < 15, d < 45, d < 90], [0, 1, 2], default=3)
    b = np.where(np.isnan(d) & diverted.to_numpy(), 3, b)
    b = np.where(cancelled.to_numpy(), 4, b)
    b = np.where(np.isnan(d) & ~cancelled.to_numpy() & ~diverted.to_numpy(), -1, b)  # unknown
    return pd.Series(b, index=delay.index)


def capped_target(delay: pd.Series, cancelled: pd.Series, diverted: pd.Series) -> pd.Series:
    """Delay with early = 0, capped at CAP; cancelled and no-delay diverted scored as CAP."""
    y = delay.clip(lower=0, upper=CAP)
    y = y.where(~(cancelled | (diverted & delay.isna())), CAP)
    return y.round()


def build(bts_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    legs = clean_legs(load_raw(bts_dir))
    flights = rdu_flights(legs)
    out_dir.mkdir(parents=True, exist_ok=True)
    legs.to_parquet(out_dir / "legs.parquet", index=False)
    flights.to_parquet(out_dir / "rdu_flights.parquet", index=False)
    return legs, flights


if __name__ == "__main__":
    import sys
    legs, flights = build(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"legs={len(legs):,} rdu_flight_rows={len(flights):,}")
