"""Download BTS On-Time Performance monthly files and keep only RDU-relevant legs.

For each month, keeps every leg flown that day by any aircraft (tail number)
that touched RDU that day. This keeps the full daily chain so we can compute
"leg number of the day" and the inbound leg's own upstream context.

Usage: python3 -I fetch_bts.py <out_dir> <start YYYY-MM> <end YYYY-MM>
"""
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"

WANT = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "IATA_CODE_Reporting_Airline", "Tail_Number",
    "Flight_Number_Reporting_Airline", "Origin", "OriginCityName", "Dest", "DestCityName",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepTimeBlk",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrTimeBlk",
    "Cancelled", "CancellationCode", "Diverted",
    "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance", "DistanceGroup",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "FirstDepTime", "TotalAddGTime", "LongestAddGTime",
]


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    out_dir = Path(sys.argv[1])
    raw_dir = out_dir / "_raw_zips"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for y, m in months(sys.argv[2], sys.argv[3]):
        target = out_dir / f"bts_rdu_chains_{y}_{m:02d}.parquet"
        if target.exists():
            print(f"skip {y}-{m:02d} (done)", flush=True)
            continue
        zpath = raw_dir / f"{y}_{m}.zip"
        r = subprocess.run(
            ["curl", "-sS", "-f", "--retry", "4", "--retry-delay", "5", "-m", "900",
             "-o", str(zpath), URL.format(y=y, m=m)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"FAIL download {y}-{m:02d}: {r.stderr.strip()}", flush=True)
            continue
        try:
            with zipfile.ZipFile(zpath) as zf:
                name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                with zf.open(name) as fh:
                    header = pd.read_csv(fh, nrows=0).columns
                cols = [c for c in WANT if c in header]
                with zf.open(name) as fh:
                    df = pd.read_csv(fh, usecols=cols, low_memory=False,
                                     dtype={"Tail_Number": "string", "CancellationCode": "string"})
        except Exception as e:  # noqa: BLE001
            print(f"FAIL parse {y}-{m:02d}: {e}", flush=True)
            continue
        rdu = (df["Origin"] == "RDU") | (df["Dest"] == "RDU")
        keys = df.loc[rdu & df["Tail_Number"].notna(), ["FlightDate", "Tail_Number"]].drop_duplicates()
        chains = df.merge(keys, on=["FlightDate", "Tail_Number"], how="inner")
        no_tail = df[rdu & df["Tail_Number"].isna()]
        keep = pd.concat([chains, no_tail], ignore_index=True)
        keep["touches_rdu"] = (keep["Origin"] == "RDU") | (keep["Dest"] == "RDU")
        keep.to_parquet(target, index=False)
        print(f"ok {y}-{m:02d}: national={len(df):,} rdu_legs={int(rdu.sum()):,} "
              f"kept_chain_legs={len(keep):,}", flush=True)
        zpath.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
