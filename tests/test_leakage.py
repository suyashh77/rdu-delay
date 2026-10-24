from datetime import date

import numpy as np
import pandas as pd

from rdu.eval import leakage
from rdu.eval.walkforward import Config, Data
from rdu.ingest.bts import clean_legs, rdu_flights
from rdu.models.baselines import MODELS


def synthetic_data(days: int = 30, tails: int = 6, seed: int = 0) -> Data:
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2023-01-01", periods=days).strftime("%Y-%m-%d"):
        for t in range(tails):
            for leg, (o, de, dep) in enumerate([("RDU", "CLT", 700), ("CLT", "RDU", 900),
                                                 ("RDU", "ATL", 1100), ("ATL", "RDU", 1400)]):
                delay = float(rng.choice([0, 5, 20, 60, 120], p=[0.6, 0.15, 0.15, 0.07, 0.03]))
                rows.append({"FlightDate": d, "Tail_Number": f"N{t}", "Reporting_Airline": "AA",
                             "Flight_Number_Reporting_Airline": 100 * t + leg, "Origin": o, "Dest": de,
                             "CRSDepTime": dep, "CRSArrTime": dep + 100, "CRSElapsedTime": 60.0,
                             "DepDelay": delay, "ArrDelay": delay - 5, "Distance": 300.0,
                             "Cancelled": float(rng.random() < 0.02), "Diverted": 0.0,
                             "CancellationCode": None, "CarrierDelay": None, "WeatherDelay": None,
                             "NASDelay": None, "SecurityDelay": None, "LateAircraftDelay": None})
    raw = pd.DataFrame(rows)
    raw["Tail_Number"] = raw["Tail_Number"].astype("string")
    raw["CancellationCode"] = raw["CancellationCode"].astype("string")
    raw["touches_rdu"] = (raw["Origin"] == "RDU") | (raw["Dest"] == "RDU")
    legs = clean_legs(raw)
    f = rdu_flights(legs)
    f = f[f["bucket"] >= 0].reset_index(drop=True)
    f["flight_id"] = f["leg_index"].astype(str) + f["direction"]
    return Data(legs, f)


CFG = Config(model="x", start="2023-01-15", end="2023-01-30", retrain_days=3)
DAYS = [date(2023, 1, 15), date(2023, 1, 25)]


def test_honest_baselines_pass_future_scramble():
    data = synthetic_data()
    for name in ["B2", "B3", "B4"]:
        r = leakage.check_future_scramble(lambda d, n=name: MODELS[n](), data, CFG, DAYS)
        assert r["passed"], name


def test_canary_is_detected():
    assert leakage.check_canary(synthetic_data(), CFG, DAYS[:1])["passed"]
