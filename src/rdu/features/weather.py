"""Flight-level forecast-weather features (EXP-002), keyed by flight_id.

For each RDU flight: conditions at RDU at its scheduled RDU time, at the other airport at its
scheduled time there, and (departures) at the inbound plane's origin at that leg's departure.
Plus day-worst values at RDU and the other airport. Every input was issued before the
22:30 ET freeze (rdu.ingest.weather); `wx_issue_max_utc` records the latest one used so the
leakage suite can verify it.
"""
import numpy as np
import pandas as pd

from rdu.eval.walkforward import ROOT, Data
from rdu.ingest import weather as W

STORE = ROOT / "data" / "features" / "weather.parquet"
TAF_COLS = ["wind_kt", "gust_kt", "vis_q", "ceil_100ft", "wx", "tempo_gust_kt", "tempo_vis_q", "tempo_wx"]
NBS_COLS = ["t03", "wsp", "gst", "cig", "vis", "psn", "pzr", "ppl", "tmp"]


def _at(points: pd.DataFrame, table: pd.DataFrame, cols: list[str], tol: str, prefix: str) -> pd.DataFrame:
    """Nearest forecast hour for each (station, forecast_date, time) point."""
    p = points.dropna(subset=["station", "time"])
    p = p.assign(time=p["time"].astype("datetime64[ns, UTC]")).sort_values("time")
    t = table.assign(valid_utc=table["valid_utc"].astype("datetime64[ns, UTC]")).sort_values("valid_utc")
    m = pd.merge_asof(p, t[["station", "forecast_date", "valid_utc", "issue_utc"] + cols],
                      left_on="time", right_on="valid_utc", by=["station", "forecast_date"],
                      direction="nearest", tolerance=pd.Timedelta(tol))
    out = m.set_index("row")[cols + ["issue_utc"]]
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out.reindex(points["row"])


def _flags(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Split TAF weather bitmasks into thunder / snow / freezing flags."""
    for src in ["wx", "tempo_wx"]:
        col = f"{prefix}_{src}"
        if col in df:
            v = df[col].astype("Float64")
            for bit, name in [(1, "ts"), (2, "sn"), (4, "fz")]:
                df[f"{prefix}_{src}_{name}"] = (v.fillna(0).astype(int) & bit).gt(0).astype(float).where(v.notna())
            df = df.drop(columns=col)
    return df


def _day_worst(table: pd.DataFrame, agg: dict, prefix: str) -> pd.DataFrame:
    g = table.groupby(["station", "forecast_date"]).agg(**agg)
    g.columns = [f"{prefix}_{c}" for c in g.columns]
    return g


def validate_issue_times(table: pd.DataFrame) -> None:
    if table.empty:
        return
    dates = pd.to_datetime(table['forecast_date'])
    cutoff = (dates - W.WEATHER_FREEZE_BEFORE_MIDNIGHT).dt.tz_localize(W.ET).dt.tz_convert('UTC')
    issued = pd.to_datetime(table['issue_utc'], utc=True)
    if (issued.isna() | (issued >= cutoff)).any():
        raise ValueError('Weather input lacks a valid pre-freeze issue timestamp')


def build(data: Data) -> pd.DataFrame:
    f = data.flights
    legs = data.legs
    taf, nbs, spc = W.load("taf"), W.load("nbs"), W.load("spc")
    for t in (taf, nbs, spc):
        t["forecast_date"] = pd.to_datetime(t["forecast_date"]).dt.date
        validate_issue_times(t)

    rdu_time = f["sched_rdu_utc"]
    other_time = np.where(f["direction"] == "dep", f["sched_arr_utc"], f["sched_dep_utc"])
    other_stn = np.where(f["direction"] == "dep", f["dest"], f["origin"])
    # inbound leg of a departure = previous row of the same aircraft-day in the legs table
    prev = legs.reindex(f["leg_index"] - 1)
    prev.index = f.index
    has_prev = (f["direction"] == "dep") & (f["leg_number"].astype(float) > 1) & prev["tail"].eq(f["tail"]).fillna(False)
    inb_time = prev["sched_dep_utc"].where(has_prev)
    inb_stn = f["prev_origin"].where(has_prev)

    base = pd.DataFrame({"row": f.index, "forecast_date": f["forecast_date"]})
    places = {
        "rdu": base.assign(station="RDU", time=rdu_time.to_numpy()),
        "oth": base.assign(station=other_stn, time=pd.to_datetime(other_time, utc=True)),
        "inb": base.assign(station=inb_stn.to_numpy(), time=pd.to_datetime(inb_time, utc=True)),
    }
    parts = []
    for name, pts in places.items():
        parts.append(_flags(_at(pts, taf, TAF_COLS, "1h", f"{name}_taf"), f"{name}_taf"))
        parts.append(_at(pts, nbs, NBS_COLS, "2h", f"{name}_nbs"))
    X = pd.concat(parts, axis=1)
    X.index = f.index

    worst = _day_worst(nbs, {"t03_max": ("t03", "max"), "gst_max": ("gst", "max"), "vis_min": ("vis", "min"),
                             "cig_min": ("cig", lambda s: s[s > 0].min()), "psn_max": ("psn", "max"),
                             "pzr_max": ("pzr", "max")}, "day")
    key_rdu = pd.MultiIndex.from_arrays([np.repeat("RDU", len(f)), f["forecast_date"]])
    key_oth = pd.MultiIndex.from_arrays([other_stn, f["forecast_date"]])
    X = X.join(worst.reindex(key_rdu).set_axis(f.index).add_prefix("rdu_"))
    X = X.join(worst.reindex(key_oth).set_axis(f.index).add_prefix("oth_"))

    spc_i = spc.set_index(["station", "forecast_date"])["spc_day2"]
    covered = set(spc["station"].unique())  # no outlook row on a covered station = no risk area (0)
    X["rdu_spc_day2"] = spc_i.reindex(key_rdu).fillna(0).to_numpy()
    X["oth_spc_day2"] = np.where(pd.Series(other_stn).isin(covered), spc_i.reindex(key_oth).fillna(0).to_numpy(), np.nan)
    # Include timestamps from day-worst and SPC features, not only sampled hours.
    for source_name, source in [('taf', taf), ('nbs', nbs), ('spc', spc)]:
        issued = source.groupby(['station', 'forecast_date'])['issue_utc'].max()
        for place_name, keys in [('rdu', key_rdu), ('oth', key_oth)]:
            X[f'{place_name}_{source_name}_source_issue_utc'] = issued.reindex(keys).to_numpy()
    X["wx_issue_max_utc"] = X[[c for c in X.columns if c.endswith("issue_utc")]].max(axis=1)
    issue_cols = [c for c in X.columns if c.endswith("_issue_utc")]
    X = X.drop(columns=issue_cols)
    X.index = f["flight_id"].to_numpy()
    return X


def load() -> pd.DataFrame:
    return pd.read_parquet(STORE)


if __name__ == "__main__":
    from rdu.eval.walkforward import load_data
    X = build(load_data())
    STORE.parent.mkdir(parents=True, exist_ok=True)
    X.astype({c: "float32" for c in X.columns if c != "wx_issue_max_utc"}).to_parquet(STORE)
    print(STORE, X.shape)
