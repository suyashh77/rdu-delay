"""Leakage tests (HANDOFF 2.8). Any failure makes an experiment invalid.

1. Future-scramble: re-run sample days after permuting every outcome not yet known at
   that day's cutoff. Predictions must not change. Covers "no BTS actuals after the
   cutoff" and "target encodings use only past dates", whatever path the model reads from.
2. Canary: a model that peeks at tomorrow's actual delay must be caught by test 1.
3. Shuffled labels: training on permuted outcomes must not beat climatology (B0).
   (HANDOFF says "no better than B1"; B1 = always on time is beaten by any calibrated
   base rate, so the meaningful bar is climatology. Logged as DL-018.)
4. Weather issued before cutoff: no weather features yet -> not applicable.
"""
from datetime import date

import numpy as np
import pandas as pd

from rdu.eval.metrics import CANCEL, GRID, summarize
from rdu.eval.walkforward import (TARGET_COLS, Config, Data, cutoff_utc, eval_days, permute_rows,
                                  predict_days)
from rdu.ingest.bts import OUTCOME_COLS
from rdu.models.baselines import B0Climatology

PRED_COLS_PREFIX = ("p_", "q")


def future_scramble(data: Data, cutoff: pd.Timestamp, seed: int = 0) -> Data:
    rng = np.random.default_rng(seed)
    future = ~(data.legs["outcome_time_utc"] < cutoff)
    # Permuting within future rows keeps every outcome_time >= cutoff, so history is unchanged.
    legs = permute_rows(data.legs, np.flatnonzero(future.to_numpy()), OUTCOME_COLS, rng)
    frows = np.flatnonzero(data.flights["leg_index"].map(future).to_numpy())
    return Data(legs, permute_rows(data.flights, frows, TARGET_COLS, rng))


def check_future_scramble(make_model, data: Data, cfg: Config, days: list[date]) -> dict:
    worst = 0.0
    for day in days:
        a = predict_days(make_model, data, cfg, [day])
        b = predict_days(make_model, future_scramble(data, cutoff_utc(day), seed=day.toordinal()), cfg, [day])
        cols = [c for c in a.columns if c.startswith(PRED_COLS_PREFIX)]
        worst = max(worst, float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max()))
    return {"test": "future_scramble", "days": [str(d) for d in days], "max_abs_change": worst,
            "passed": worst < 1e-9}


class Canary:
    """Deliberate cheat: puts 50% of the mass on the flight's actual outcome."""

    def __init__(self, data: Data):
        self.data, self.base = data, B0Climatology()

    def fit(self, hist):
        self.base.fit(hist)

    def predict(self, targets, sched, hist):
        truth = self.data.flights.loc[targets.index]
        y = np.where(truth["bucket"] == 4, CANCEL, truth["y_capped"].fillna(180)).astype(int)
        return 0.5 * self.base.predict(targets, sched, hist) + 0.5 * np.eye(GRID)[y]


def check_canary(data: Data, cfg: Config, days: list[date]) -> dict:
    r = check_future_scramble(Canary, data, cfg, days)
    return {"test": "canary_detected", "max_abs_change": r["max_abs_change"], "passed": not r["passed"]}


def check_shuffled(make_model, data: Data, cfg: Config, tol: float = 0.001) -> dict:
    days = eval_days(cfg)
    shuffled = summarize(predict_days(make_model, data, Config(**{**cfg.__dict__, "shuffle_labels": True}), days))
    clim = summarize(predict_days(lambda d: B0Climatology(), data, cfg, days))
    return {"test": "shuffled_labels", "brier_shuffled": shuffled["brier"], "brier_climatology": clim["brier"],
            "passed": shuffled["brier"] >= clim["brier"] - tol}


def check_weather_issue_time(data: Data) -> dict:
    """Verify that every weather feature row was issued before that day's 23:00 ET cutoff."""
    from rdu.features import weather as W
    try:
        wx = W.load()
    except Exception as e:
        return {"test": "weather_issue_time", "passed": False, "error": str(e)}
    u_dates = pd.Series(data.flights["forecast_date"].unique())
    date_map = dict(zip(u_dates, u_dates.map(cutoff_utc)))
    cutoffs = data.flights["forecast_date"].map(date_map).to_numpy()
    max_issue = wx.loc[data.flights["flight_id"], "wx_issue_max_utc"].to_numpy()
    valid = pd.Series(max_issue).notna().to_numpy()
    late = int((max_issue[valid] > cutoffs[valid]).sum())
    return {"test": "weather_issue_time", "passed": late == 0, "n_checked": int(valid.sum()), "n_late": late}


def run_all(make_model, data: Data, cfg: Config, n_days: int = 4) -> dict:
    """Fast-mode settings: sample days spread over the window, shuffled test on stride-28 days."""
    all_days = eval_days(cfg)
    sample = [all_days[i] for i in np.linspace(0, len(all_days) - 1, n_days).astype(int)]
    fast = Config(**{**cfg.__dict__, "stride": 28})
    results = [check_future_scramble(make_model, data, cfg, sample),
               check_canary(data, cfg, sample[:1]),
               check_shuffled(make_model, data, fast),
               check_weather_issue_time(data)]
    return {"passed": all(r["passed"] for r in results), "results": results}

