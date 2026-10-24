import pandas as pd

from rdu.ingest.bts import bucket_of, capped_target, clean_legs, rdu_flights


def _raw(rows):
    base = {"Reporting_Airline": "AA", "Distance": 500.0, "Cancelled": 0.0, "Diverted": 0.0,
            "CancellationCode": None, "CarrierDelay": None, "WeatherDelay": None, "NASDelay": None,
            "SecurityDelay": None, "LateAircraftDelay": None}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["Tail_Number"] = df["Tail_Number"].astype("string")
    df["touches_rdu"] = (df["Origin"] == "RDU") | (df["Dest"] == "RDU")
    return df


def test_red_eye_crosses_midnight_and_timezones():
    # LAX 23:30 PDT (UTC-7) -> RDU, 5h elapsed => 06:30 UTC + 5h = 11:30 UTC = 07:30 EDT next day
    raw = _raw([{"FlightDate": "2022-06-01", "Tail_Number": "N1", "Flight_Number_Reporting_Airline": 1,
                 "Origin": "LAX", "Dest": "RDU", "CRSDepTime": 2330, "CRSArrTime": 730,
                 "CRSElapsedTime": 300.0, "DepDelay": 10.0, "ArrDelay": 20.0}])
    legs = clean_legs(raw)
    assert legs.loc[0, "sched_arr_utc"] == pd.Timestamp("2022-06-02 11:30", tz="UTC")
    assert legs.loc[0, "actual_arr_utc"] == pd.Timestamp("2022-06-02 11:50", tz="UTC")
    f = rdu_flights(legs)
    assert str(f.loc[0, "forecast_date"]) == "2022-06-02"  # RDU local arrival date


def test_leg_numbers_and_turns():
    raw = _raw([
        {"FlightDate": "2022-06-01", "Tail_Number": "N1", "Flight_Number_Reporting_Airline": 2,
         "Origin": "RDU", "Dest": "CLT", "CRSDepTime": 1000, "CRSArrTime": 1100, "CRSElapsedTime": 60.0,
         "DepDelay": 0.0, "ArrDelay": 0.0},
        {"FlightDate": "2022-06-01", "Tail_Number": "N1", "Flight_Number_Reporting_Airline": 1,
         "Origin": "CLT", "Dest": "RDU", "CRSDepTime": 700, "CRSArrTime": 800, "CRSElapsedTime": 60.0,
         "DepDelay": 0.0, "ArrDelay": 0.0},
    ])
    legs = clean_legs(raw).set_index("flight_number")
    assert legs.loc[1, "leg_number"] == 1 and legs.loc[2, "leg_number"] == 2
    assert legs.loc[2, "sched_turn_min"] == 120
    assert not legs.loc[2, "chain_break"]


def test_buckets_and_capped_target():
    delay = pd.Series([-5.0, 15.0, 44.0, 45.0, 300.0, None, None, None])
    canc = pd.Series([False] * 5 + [True, False, False])
    div = pd.Series([False] * 6 + [True, False])
    assert bucket_of(delay, canc, div).tolist() == [0, 1, 1, 2, 3, 4, 3, -1]
    y = capped_target(delay, canc, div)
    assert y.tolist()[:7] == [0, 15, 44, 45, 180, 180, 180]
