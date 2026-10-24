from datetime import date

import pandas as pd

from rdu.ingest.weather import condense_spc, condense_taf, cutoffs


def _taf_rows(issue, rows):
    pid = f"{issue}-TAF"
    base = {"station": "KRDU", "valid": issue, "is_tempo": False, "fx_valid_end": None, "drct": 90.0,
            "gust": 0.0, "presentwx": "[]", "skyc": "[]", "skyl": "[]", "product_id": pid, "ftype": "Forecast"}
    return [{**base, **r} for r in rows]


def test_taf_uses_last_issue_before_cutoff_and_tempo_overlay():
    # Weather freeze for 2024-03-02 is 2024-03-02 03:30Z (22:30 EST on 03-01).
    early = _taf_rows("2024-03-01 23:30", [
        {"fx_valid": "2024-03-01 23:30", "sknt": 5.0, "visibility": 6.01},
        {"fx_valid": "2024-03-02 15:00", "sknt": 12.0, "visibility": 3.0, "skyc": "['BKN']", "skyl": "[800]",
         "presentwx": "['-RA']"},
        {"fx_valid": "2024-03-02 18:00", "fx_valid_end": "2024-03-02 21:00", "is_tempo": True, "sknt": None,
         "visibility": 1.0, "gust": 35.0, "presentwx": "['TSRA']"},
    ])
    late = _taf_rows("2024-03-02 03:40", [  # issued after the cutoff: must be ignored
        {"fx_valid": "2024-03-02 03:40", "sknt": 40.0, "visibility": 0.25},
    ])
    out = condense_taf(pd.DataFrame(early + late), cutoffs(date(2024, 3, 2), date(2024, 3, 2)))
    assert (out["issue_utc"] == pd.Timestamp("2024-03-01 23:30", tz="UTC")).all()
    at = out.set_index("valid_utc")
    h14, h19 = pd.Timestamp("2024-03-02 14:30", tz="UTC"), pd.Timestamp("2024-03-02 19:30", tz="UTC")
    assert at.loc[h14, "wind_kt"] == 5 and at.loc[h14, "vis_q"] == 24
    assert at.loc[h19, "wind_kt"] == 12 and at.loc[h19, "ceil_100ft"] == 8 and at.loc[h19, "vis_q"] == 12
    assert at.loc[h19, "tempo_gust_kt"] == 35 and at.loc[h19, "tempo_wx"] & 1  # thunder in TEMPO
    assert pd.isna(at.loc[h14, "tempo_wx"])
    # 24-hour TAF: hours past issue + 24 h are unknown, not filled
    assert at.loc[pd.Timestamp("2024-03-03 00:30", tz="UTC"), "wind_kt"] is pd.NA


def test_spc_keeps_latest_issue_before_cutoff():
    outlooks = [
        {"utc_issue": "2024-05-10T12:00:00Z", "utc_product_issue": "2024-05-09T06:00:00Z", "threshold": "MRGL"},
        {"utc_issue": "2024-05-10T12:00:00Z", "utc_product_issue": "2024-05-09T17:30:00Z", "threshold": "SLGT"},
        {"utc_issue": "2024-05-11T12:00:00Z", "utc_product_issue": "2024-05-11T06:00:00Z", "threshold": "ENH"},
    ]  # the last one is issued on the day itself, after the cutoff
    out = condense_spc(outlooks, cutoffs(date(2024, 5, 10), date(2024, 5, 11)))
    assert out["forecast_date"].tolist() == [date(2024, 5, 10)]
    assert out["spc_day2"].tolist() == [3]
