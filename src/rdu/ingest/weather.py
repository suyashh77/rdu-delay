"""Forecast weather (never observations) from the Iowa Environmental Mesonet, stored small.

Per station and forecast day D (RDU local), keep only what was issued before 23:00 ET on D-1
and only the hours RDU flights use (D 05:00 ET to D+1 03:00 ET). Raw downloads are condensed
in memory and discarded; nothing but the compact tables below is written.

    taf/<STN>.parquet  hourly prevailing + TEMPO/PROB conditions from the last TAF before cutoff
    nbs/<STN>.parquet  NBM station forecast (3-hourly) from the 01Z run on D's UTC date
    spc/<STN>.parquet  SPC Day 2 categorical risk at the airport, latest issued before cutoff

Usage: python -m rdu.ingest.weather <taf|nbs|spc|all> [STN ...]
"""
import ast
import http.client
import io
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "weather"
ET = "America/New_York"
START, END = date(2022, 1, 1), date(2026, 8, 31)
IEM = "https://mesonet.agron.iastate.edu"
# RDU + the 37 airports covering 99% of RDU flights' other ends and inbound origins (2022-01..2026-08).
STATIONS = ["RDU", "ATL", "LGA", "JFK", "CLT", "BOS", "ORD", "MCO", "DCA", "EWR", "PHL", "MIA", "BNA",
            "DFW", "DEN", "BWI", "DTW", "TPA", "FLL", "MSP", "AUS", "IAD", "MDW", "IAH", "SEA", "PIT",
            "LAX", "SFO", "STL", "SJU", "PHX", "LAS", "DAL", "SLC", "HOU", "CVG", "CLE", "BUF"]
WINDOW_START_H, WINDOW_END_H = 7, 29   # hours after the 22:30 ET freeze: D 05:30 ET .. D+1 03:30 ET
NBS_RUN_HOUR_UTC = 1                    # 01Z run, out ~1 h before a 03Z (EDT) / 04Z (EST) cutoff
NBS_KEEP = ["tmp", "wdr", "wsp", "gst", "cig", "vis", "sky", "t03", "p06", "pra", "psn", "pzr", "ppl",
            "s06", "i06"]               # airport-relevant only: wind, ceiling, visibility, storms, precip type
POLITE_S, BACKOFF_S = 6, 90             # IEM rate-limits bursts


def icao(stn: str) -> str:
    return "TJSJ" if stn == "SJU" else f"K{stn}"


# Weather inputs freeze at 22:30 ET, when the nightly job starts (HANDOFF §9), not 23:00:
# some TAF amendments land minutes before 23:00 and production would not see them.
WEATHER_FREEZE_BEFORE_MIDNIGHT = pd.Timedelta(minutes=90)


def cutoffs(start: date = START, end: date = END) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    cut = (days - WEATHER_FREEZE_BEFORE_MIDNIGHT).tz_localize(ET).tz_convert("UTC")  # 22:30 ET on D-1
    return pd.DataFrame({"forecast_date": days.date, "cutoff_utc": cut})


def fetch(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "rdu-delay research"}),
                                        timeout=600) as r:
                body = r.read()
            if body.startswith(b"Too many requests"):
                raise urllib.error.HTTPError(url, 429, "rate limited", None, None)
            time.sleep(POLITE_S)
            return body
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                ConnectionError) as e:
            wait = BACKOFF_S * (i + 1)
            print(f"  retry {i + 1}/{tries} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up: {url}")


# ---------- TAF ----------

def _lst(s) -> list:
    return ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []


def _ceiling(skyc: str, skyl: str) -> float:
    """Lowest BKN/OVC/VV layer in hundreds of feet; NaN if none."""
    heights = [h for c, h in zip(_lst(skyc), _lst(skyl)) if c in ("BKN", "OVC", "VV") and h is not None]
    return min(heights) / 100 if heights else np.nan


def _wx_flags(wx: str) -> int:
    """Bitmask: 1 thunder, 2 snow, 4 freezing precip/ice pellets, 8 heavy precip, 16 fog."""
    codes = " ".join(_lst(wx))
    return (1 * ("TS" in codes) + 2 * ("SN" in codes) + 4 * ("FZ" in codes or "PL" in codes)
            + 8 * ("+" in codes) + 16 * ("FG" in codes))


def condense_taf(raw: pd.DataFrame, cut: pd.DataFrame) -> pd.DataFrame:
    """Hourly conditions for each forecast day from the last TAF issued before its cutoff."""
    t = raw.copy()
    t["issue_utc"] = pd.to_datetime(t["valid"], utc=True)
    t["start"] = pd.to_datetime(t["fx_valid"], utc=True)
    t["end"] = pd.to_datetime(t["fx_valid_end"], utc=True)
    t["temp"] = t["end"].notna() | t["is_tempo"].astype(str).eq("True")
    t["ceil"] = [_ceiling(c, l) for c, l in zip(t["skyc"], t["skyl"])]
    t["wx"] = [_wx_flags(w) for w in t["presentwx"]]
    products = t.groupby("product_id")["issue_utc"].first().sort_values().reset_index()
    pick = pd.merge_asof(cut.sort_values("cutoff_utc"), products, left_on="cutoff_utc", right_on="issue_utc",
                         allow_exact_matches=False).dropna(subset=["product_id"])
    hours = np.arange(WINDOW_START_H, WINDOW_END_H)
    grid = pick.loc[pick.index.repeat(len(hours))].reset_index(drop=True)
    grid["valid_utc"] = grid["cutoff_utc"] + pd.to_timedelta(np.tile(hours, len(pick)), unit="h")
    grid = grid.drop(columns="issue_utc")
    base = t[~t["temp"]].sort_values("start")
    # TAF horizon: 30 h at stations that issue 30-hour TAFs, else 24 h
    horizon = 30 if (t["start"] - t["issue_utc"]).max() > pd.Timedelta(hours=24) else 24
    prevailing = pd.merge_asof(grid.sort_values("valid_utc"), base[["product_id", "issue_utc", "start", "sknt",
                                                                    "drct", "gust", "visibility", "ceil", "wx"]],
                               left_on="valid_utc", right_on="start", by="product_id")
    prevailing.loc[prevailing["valid_utc"] >= prevailing["issue_utc"] + pd.Timedelta(hours=horizon),
                   ["sknt", "drct", "gust", "visibility", "ceil", "wx"]] = np.nan
    temp = t[t["temp"]][["product_id", "start", "end", "gust", "visibility", "ceil", "wx"]]
    over = prevailing[["product_id", "valid_utc"]].merge(temp, on="product_id")
    over = over[(over["valid_utc"] >= over["start"]) & (over["valid_utc"] < over["end"])]
    tg = over.groupby(["product_id", "valid_utc"]).agg(tempo_gust=("gust", "max"), tempo_vis=("visibility", "min"),
                                                       tempo_ceil=("ceil", "min"),
                                                       tempo_wx=("wx", lambda s: int(np.bitwise_or.reduce(s.astype(int)))))
    out = prevailing.merge(tg.reset_index(), on=["product_id", "valid_utc"], how="left")
    return pd.DataFrame({
        "forecast_date": out["forecast_date"],
        "valid_utc": out["valid_utc"],
        "issue_utc": out["issue_utc"],
        "wind_kt": out["sknt"].astype("Float32").round().astype("Int16"),
        "wind_dir": out["drct"].astype("Float32").round().astype("Int16"),
        "gust_kt": out["gust"].astype("Float32").round().astype("Int16"),
        "vis_q": (out["visibility"] * 4).round().clip(upper=25).astype("Int8"),  # quarter-miles, 25 = 6+ mi
        "ceil_100ft": out["ceil"].round().clip(upper=999).astype("Int16"),
        "wx": out["wx"].astype("Int8"),
        "tempo_gust_kt": out["tempo_gust"].round().astype("Int16"),
        "tempo_vis_q": (out["tempo_vis"] * 4).round().clip(upper=25).astype("Int8"),
        "tempo_ceil_100ft": out["tempo_ceil"].round().clip(upper=999).astype("Int16"),
        "tempo_wx": out["tempo_wx"].astype("Int8"),
    })


def taf_station(stn: str) -> Path:
    path = OUT / "taf" / f"{stn}.parquet"
    if path.exists():
        return path
    parts = []
    for y in range(START.year, END.year + 1):
        # pad a day either side so the TAF before each cutoff is present
        s = max(date(y, 1, 1), START) - timedelta(days=2)
        e = min(date(y, 12, 31), END) + timedelta(days=1)
        url = f"{IEM}/cgi-bin/request/taf.py?station={icao(stn)}&sts={s}T00:00Z&ets={e}T00:00Z&fmt=csv"
        raw = pd.read_csv(io.BytesIO(fetch(url)))
        cut = cutoffs(max(date(y, 1, 1), START), min(date(y, 12, 31), END))
        if not raw.empty:
            parts.append(condense_taf(raw, cut))
        print(f"  taf {stn} {y}: raw {len(raw):,} rows", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


# ---------- NBM station forecasts (NBS) ----------

def condense_nbs(raw: pd.DataFrame, cut: pd.DataFrame) -> pd.DataFrame:
    r = raw.copy()
    r["runtime"] = pd.to_datetime(r["runtime"], utc=True)
    r["valid_utc"] = pd.to_datetime(r["ftime"], utc=True)
    r = r[r["runtime"].dt.hour == NBS_RUN_HOUR_UTC]
    c = cut.copy()
    c["runtime"] = (pd.to_datetime(c["cutoff_utc"]).dt.floor("D") + pd.Timedelta(hours=NBS_RUN_HOUR_UTC))
    if not (c["runtime"] < c["cutoff_utc"]).all():
        raise ValueError("NBS run would be after the cutoff")
    m = c.merge(r, on="runtime")
    lead = m["valid_utc"] - m["cutoff_utc"]
    m = m[(lead >= pd.Timedelta(hours=WINDOW_START_H - 3)) & (lead < pd.Timedelta(hours=WINDOW_END_H))]
    out = m[["forecast_date", "valid_utc", "runtime"]].rename(columns={"runtime": "issue_utc"})
    for col in NBS_KEEP:
        out[col] = pd.to_numeric(m[col], errors="coerce").round().astype("Int16")
    return out.reset_index(drop=True)


def nbs_station(stn: str) -> Path:
    path = OUT / "nbs" / f"{stn}.parquet"
    if path.exists():
        return path
    parts = []
    for y in range(START.year, END.year + 1):
        s, e = max(date(y, 1, 1), START), min(date(y, 12, 31), END) + timedelta(days=1)
        url = f"{IEM}/cgi-bin/request/mos.py?station={icao(stn)}&model=NBS&sts={s}T00:00Z&ets={e}T03:00Z&format=csv"
        raw = pd.read_csv(io.BytesIO(fetch(url)))
        if not raw.empty:
            parts.append(condense_nbs(raw, cutoffs(s, e - timedelta(days=1))))
        print(f"  nbs {stn} {y}: raw {len(raw):,} rows", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


# ---------- SPC Day 2 categorical outlook at the airport ----------

SPC_LEVEL = {"TSTM": 1, "MRGL": 2, "SLGT": 3, "ENH": 4, "MDT": 5, "HIGH": 6}


def condense_spc(outlooks: list[dict], cut: pd.DataFrame) -> pd.DataFrame:
    if not outlooks:
        return pd.DataFrame(columns=["forecast_date", "issue_utc", "spc_day2"])
    o = pd.DataFrame(outlooks)
    o["issue_utc"] = pd.to_datetime(o["utc_product_issue"], utc=True)
    o["valid_start"] = pd.to_datetime(o["utc_issue"], utc=True)  # 12Z on the outlook day
    o["forecast_date"] = (o["valid_start"] - pd.Timedelta(hours=12)).dt.date
    o["spc_day2"] = o["threshold"].map(SPC_LEVEL).astype("Int8")
    m = o.merge(cut, on="forecast_date")
    m = m[m["issue_utc"] < m["cutoff_utc"]]
    latest = m.sort_values("issue_utc").groupby("forecast_date").tail(1)
    return latest[["forecast_date", "issue_utc", "spc_day2"]].reset_index(drop=True)


def spc_station(stn: str) -> Path:
    path = OUT / "spc" / f"{stn}.parquet"
    if path.exists():
        return path
    a = airportsdata.load("IATA")[stn]
    url = f"{IEM}/json/spcoutlook.py?lon={a['lon']:.4f}&lat={a['lat']:.4f}&last=5000&day=2&cat=categorical"
    outlooks = json.loads(fetch(url))["outlooks"]
    df = condense_spc(outlooks, cutoffs())
    print(f"  spc {stn}: {len(outlooks)} outlooks, {len(df)} days with risk", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load(source: str) -> pd.DataFrame:
    frames = [pd.read_parquet(p).assign(station=p.stem) for p in sorted((OUT / source).glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    which = sys.argv[1]
    stations = sys.argv[2:] or STATIONS
    jobs = {"taf": taf_station, "nbs": nbs_station, "spc": spc_station}
    for src in (jobs if which == "all" else [which]):
        for stn in stations:
            try:
                print(jobs[src](stn), flush=True)
            except Exception as e:  # log and continue; never fill missing data
                print(f"FAIL {src} {stn}: {e}", flush=True)
