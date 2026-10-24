import numpy as np
import pandas as pd

from rdu.eval.metrics import (CANCEL, GRID, block_bootstrap_diff, bucket_probs, ece, quantiles,
                              score, summaries)


def point_mass(k, n=1):
    P = np.zeros((n, GRID))
    P[:, k] = 1.0
    return P


def test_crps_of_point_mass_is_absolute_error():
    s = score(point_mass(10), np.array([40]), np.array([1]))
    assert s.loc[0, "crps"] == 30
    # cancellation is scored as 180
    s = score(point_mass(CANCEL), np.array([0]), np.array([0]))
    assert s.loc[0, "crps"] == 180


def test_crps_uniform_matches_closed_form():
    # Uniform over 0..9, outcome 0: sum_{k=0}^{8} ((k+1)/10)^2 = 2.85
    P = np.zeros((1, GRID))
    P[0, :10] = 0.1
    assert np.isclose(score(P, np.array([0]), np.array([0])).loc[0, "crps"], 2.85)


def test_buckets_brier_quantiles():
    P = np.zeros((1, GRID))
    P[0, 0], P[0, 20], P[0, 100], P[0, CANCEL] = 0.6, 0.2, 0.1, 0.1
    assert np.allclose(bucket_probs(P), [[0.6, 0.2, 0.0, 0.1, 0.1]])
    s = score(P, np.array([20]), np.array([1]))
    assert np.isclose(s.loc[0, "brier"], (0.3 - 1) ** 2)
    assert np.isclose(s.loc[0, "logloss"], -np.log(0.2))
    assert quantiles(P, [0.5, 0.7, 0.95]).tolist() == [[0, 20, 180]]
    out = summaries(P)
    assert np.isclose(out.loc[0, "expected_delay_capped_min"], (20 * 0.2 + 100 * 0.1) / 0.9)
    assert np.isclose(out.loc[0, "typical_if_delayed_min"], (20 * 0.2 + 100 * 0.1) / 0.3)


def test_ece_perfectly_calibrated_is_zero():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y) == 0


def test_bootstrap_identical_runs_and_clear_win():
    days = np.repeat(pd.date_range("2023-01-01", periods=60).date, 10)
    a = pd.DataFrame({"forecast_date": days, "flight_id": range(600), "brier": 0.1})
    r = block_bootstrap_diff(a, a.copy())
    assert r["diff"] == 0 and not r["excludes_zero"]
    rng = np.random.default_rng(1)
    b = a.assign(brier=0.2 + rng.normal(0, 0.01, 600))
    r = block_bootstrap_diff(a, b)
    assert r["diff"] < 0 and r["excludes_zero"]


def test_pit_coverage_is_80_for_calibrated_lumpy_forecast():
    # True distribution: 78% at 0, rest uniform on 1..50. Forecast = truth.
    rng = np.random.default_rng(0)
    P = np.zeros((1, GRID))
    P[0, 0] = 0.78
    P[0, 1:51] = 0.22 / 50
    y = np.where(rng.random(20000) < 0.78, 0, rng.integers(1, 51, 20000))
    s = score(np.repeat(P, len(y), 0), y, np.zeros(len(y), int))
    assert abs(s["in80"].mean() - 0.80) < 0.01
    assert s["in80_naive"].mean() > 0.88  # why the naive check is unusable here


def test_slice_columns_handles_missing_leg_number():
    from rdu.eval.metrics import slice_columns
    f = pd.DataFrame({"rdu_local_hour": [6, 16], "leg_number": pd.array([1, None], dtype="Int64"),
                      "carrier": ["AA", "YX"], "forecast_date": ["2023-01-01", "2023-07-01"],
                      "direction": ["dep", "arr"]})
    s = slice_columns(f)
    assert s["leg_group"].tolist() == ["1", "unknown"]
    assert s["carrier_type"].tolist() == ["mainline", "regional"]
