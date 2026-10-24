"""Walk-forward harness (HANDOFF 2.5).

For each forecast day D: cutoff = 23:00 ET on D-1. History = legs whose outcome was
known before the cutoff. Targets and schedule are passed with outcome columns removed.
Models are refit every `retrain_days`; between refits only `hist` passed to predict updates.
"""
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rdu.eval.metrics import (block_bootstrap_diff, score, slice_columns, slice_table, summaries,
                              summarize, target_bin)
from rdu.ingest.bts import ET, OUTCOME_COLS
from rdu.models.baselines import History

ROOT = Path(__file__).resolve().parents[3]
TARGET_COLS = ["delay_min", "bucket", "y_capped"]


@dataclass
class Config:
    model: str
    start: str = "2023-01-01"
    end: str = "2025-06-30"  # dev period; holdout starts 2025-07-01 (HANDOFF 2.4)
    stride: int = 1  # 1 = full mode; 7 = fast mode
    retrain_days: int = 7
    shuffle_labels: bool = False
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Data:
    legs: pd.DataFrame
    flights: pd.DataFrame


def load_data(clean_dir: Path = ROOT / "data" / "clean") -> Data:
    legs = pd.read_parquet(clean_dir / "legs.parquet")
    flights = pd.read_parquet(clean_dir / "rdu_flights.parquet")
    flights = flights[flights["bucket"] >= 0].reset_index(drop=True)  # unknown outcome: not scorable
    flights["flight_id"] = flights["leg_index"].astype(str) + flights["direction"]
    return Data(legs, flights)


def cutoff_utc(day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day - timedelta(days=1), datetime.min.time())
                        ).replace(hour=23).tz_localize(ET).tz_convert("UTC")


def history(data: Data, cutoff: pd.Timestamp) -> History:
    known = data.legs["outcome_time_utc"] < cutoff
    return History(data.legs[known], data.flights[data.flights["leg_index"].map(known)])


def schedule_view(data: Data, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Targets for day D and the full aircraft-days around them, schedule columns only."""
    targets = data.flights[data.flights["forecast_date"] == day]
    days = {day - timedelta(days=1), day}
    sched = data.legs[data.legs["flight_date"].isin(days)]
    return (targets.drop(columns=[c for c in OUTCOME_COLS + TARGET_COLS if c in targets]),
            sched.drop(columns=OUTCOME_COLS))


def permute_rows(df: pd.DataFrame, rows: np.ndarray, cols: list[str], rng) -> pd.DataFrame:
    """Copy of df with `cols` jointly permuted among positional `rows`; dtypes kept."""
    out = df.copy()
    src = rows[rng.permutation(len(rows))]
    for c in cols:
        vals = out[c].to_numpy().copy()
        vals[rows] = vals[src]
        out[c] = pd.Series(vals, index=out.index, dtype=df[c].dtype)
    return out


def shuffle_history(hist: History, seed: int) -> History:
    """Permute outcomes across rows: destroys any link between inputs and labels."""
    rng = np.random.default_rng(seed)
    legs = permute_rows(hist.legs, np.arange(len(hist.legs)), OUTCOME_COLS, rng)
    f = permute_rows(hist.flights, np.arange(len(hist.flights)), TARGET_COLS, rng)
    return History(legs, f)


def eval_days(cfg: Config) -> list[date]:
    days = pd.date_range(cfg.start, cfg.end).date
    return list(days[::cfg.stride])


def predict_days(make_model: Callable, data: Data, cfg: Config, days: list[date]) -> pd.DataFrame:
    """Run the walk-forward loop; returns per-flight predictions + scores."""
    model, last_fit, out = None, None, []
    for day in days:
        cut = cutoff_utc(day)
        hist = history(data, cut)
        if cfg.shuffle_labels:
            hist = shuffle_history(hist, seed=day.toordinal())
        if model is None or (day - last_fit).days >= cfg.retrain_days:
            model = make_model(data)
            model.fit(hist)
            last_fit = day
        targets, sched = schedule_view(data, day)
        if targets.empty:
            continue
        P = model.predict(targets, sched, hist)
        if P.shape != (len(targets), 182) or not np.allclose(P.sum(1), 1, atol=1e-6) or (P < 0).any():
            raise ValueError(f"bad prediction matrix on {day}")
        truth = data.flights.loc[targets.index]
        s = score(P, target_bin(truth["y_capped"], truth["bucket"]), truth["bucket"].to_numpy())
        s.index = targets.index
        out.append(pd.concat([truth[["flight_id", "forecast_date"]], summaries(P).set_index(targets.index), s], axis=1))
    return pd.concat(out)


def git_hash() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True)
    return (r.stdout.strip() or "nocommit") + ("-dirty" if dirty.stdout.strip() else "")


def run(make_model: Callable, cfg: Config, data: Data, run_id: str) -> dict:
    t0 = time.time()
    preds = predict_days(make_model, data, cfg, eval_days(cfg))
    out = ROOT / "experiments" / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "predictions.parquet")
    day = preds.groupby("forecast_date")[["brier", "crps", "logloss"]].mean()
    day.to_csv(out / "day_scores.csv")
    sl = slice_table(preds, slice_columns(data.flights.loc[preds.index]))
    sl.to_csv(out / "slices.csv", index=False)
    summary = {"run_id": run_id, **summarize(preds), "config": asdict(cfg), "git": git_hash(),
               "runtime_s": round(time.time() - t0, 1), "n_days": int(preds["forecast_date"].nunique())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def compare(run_a: str, run_b: str) -> dict:
    """Acceptance-rule numbers for run_a (challenger) vs run_b (champion)."""
    base = ROOT / "experiments" / "results"
    a = pd.read_parquet(base / run_a / "predictions.parquet")
    b = pd.read_parquet(base / run_b / "predictions.parquet")
    sa, sb = summarize(a), summarize(b)
    return {
        "brier_skill": 1 - sa["brier"] / sb["brier"],
        "crps_skill": 1 - sa["crps"] / sb["crps"],
        "bootstrap_brier": block_bootstrap_diff(a, b, "brier"),
        "bootstrap_crps": block_bootstrap_diff(a, b, "crps"),
        "ece_change": sa["ece"] - sb["ece"],
        "logloss_change": sa["logloss"] - sb["logloss"],
        "coverage80": sa["coverage80"],
    }

