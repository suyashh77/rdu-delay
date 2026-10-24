"""Scoring for distributional forecasts.

Every model outputs P with shape (n, GRID): columns 0..CAP are the probability that
the capped delay (early = 0) equals that many minutes, column CAP+1 is cancelled.
Rows sum to 1.
"""
import numpy as np
import pandas as pd

from rdu.ingest.bts import BUCKETS, CAP

GRID = CAP + 2
CANCEL = CAP + 1
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
EPS = 1e-6  # floor for log loss so a hard-zero forecast (B1) scores finite but terrible
REGIONAL = {"9E", "MQ", "OH", "OO", "YV", "YX", "QX", "EV", "C5", "G7", "ZW", "PT"}


def target_bin(y_capped: pd.Series, bucket: pd.Series) -> np.ndarray:
    """Grid index of the outcome: capped minutes, or CANCEL."""
    return np.where(bucket.to_numpy() == 4, CANCEL, y_capped.fillna(CAP).to_numpy()).astype(int)


def bucket_probs(P: np.ndarray) -> np.ndarray:
    return np.stack([P[:, 0:15].sum(1), P[:, 15:45].sum(1), P[:, 45:90].sum(1),
                     P[:, 90:CAP + 1].sum(1), P[:, CANCEL]], axis=1)


def _crps_cdf(P: np.ndarray) -> np.ndarray:
    """CDF on 0..CAP with cancellations scored as CAP."""
    Q = P[:, :CAP + 1].copy()
    Q[:, CAP] += P[:, CANCEL]
    return np.cumsum(Q, axis=1)


def quantiles(P: np.ndarray, qs=QUANTILES) -> np.ndarray:
    F = _crps_cdf(P)
    return np.stack([(F < q - 1e-12).sum(1) for q in qs], axis=1)


def summaries(P: np.ndarray) -> pd.DataFrame:
    """Per-flight published fields (HANDOFF 2.3)."""
    b = bucket_probs(P)
    k = np.arange(CAP + 1)
    flown = P[:, :CAP + 1]
    late = flown[:, 15:]
    out = pd.DataFrame(b, columns=[f"p_{n}" for n in BUCKETS])
    out["p_delay15"] = b[:, 1:4].sum(1)
    out["expected_delay_capped_min"] = (flown * k).sum(1) / np.maximum(flown.sum(1), EPS)
    out["typical_if_delayed_min"] = (late * k[15:]).sum(1) / np.maximum(late.sum(1), EPS)
    for q, col in zip(QUANTILES, quantiles(P).T):
        out[f"q{int(round(q * 100)):02d}"] = col
    return out


def score(P: np.ndarray, y_bin: np.ndarray, bucket: np.ndarray) -> pd.DataFrame:
    """Per-flight scores. Averages of these are the run metrics."""
    b = bucket_probs(P)
    p15 = b[:, 1:4].sum(1)
    y15 = np.isin(bucket, [1, 2, 3]).astype(float)
    y_crps = np.minimum(y_bin, CAP)
    Fall = _crps_cdf(P)
    F = Fall[:, :CAP]  # integrate over [0, CAP)
    step = (np.arange(CAP)[None, :] >= y_crps[:, None]).astype(float)
    q = quantiles(P, [0.10, 0.90])
    # Randomized-PIT coverage (DL-017): the outcome's CDF step [F(y-), F(y)] is spread
    # uniformly; credit = share of it inside [0.10, 0.90]. Calibrated forecast -> exactly 0.80,
    # even with the big point mass at 0 that makes the naive [q10, q90] check read >= 0.90.
    r = np.arange(len(P))
    hi = Fall[r, y_crps]
    lo = np.where(y_crps > 0, Fall[r, np.maximum(y_crps - 1, 0)], 0.0)
    overlap = np.clip(np.minimum(hi, 0.9) - np.maximum(lo, 0.1), 0, None)
    width = hi - lo
    point_inside = ((hi >= 0.1) & (hi <= 0.9)).astype(float)
    in80 = np.where(width > 1e-12, overlap / np.maximum(width, 1e-12), point_inside)
    return pd.DataFrame({
        "p15": p15,
        "y15": y15,
        "brier": (p15 - y15) ** 2,
        "crps": ((F - step) ** 2).sum(1),
        "logloss": -np.log(np.maximum(b[np.arange(len(b)), bucket], EPS)),
        "in80": in80,
        "in80_naive": ((y_crps >= q[:, 0]) & (y_crps <= q[:, 1])).astype(float),
    })


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for i in range(bins):
        m = idx == i
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return total / len(p)


def summarize(s: pd.DataFrame) -> dict:
    return {
        "n": int(len(s)),
        "brier": float(s["brier"].mean()),
        "crps": float(s["crps"].mean()),
        "logloss": float(s["logloss"].mean()),
        "ece": ece(s["p15"].to_numpy(), s["y15"].to_numpy()),
        "coverage80": float(s["in80"].mean()),
        "coverage80_naive": float(s["in80_naive"].mean()),
        "base_rate15": float(s["y15"].mean()),
    }


def slice_columns(f: pd.DataFrame) -> pd.DataFrame:
    """Slice labels for each scored flight (HANDOFF 2.6). Storm-day slice needs SPC/NWS data: not yet."""
    h = f["rdu_local_hour"]
    leg = f["leg_number"].astype(float).to_numpy()  # NaN when tail unknown
    return pd.DataFrame({
        "hour_block": pd.cut(h, [-1, 4, 8, 11, 14, 17, 20, 23],
                             labels=["00-04", "05-08", "09-11", "12-14", "15-17", "18-20", "21-23"]).astype(str),
        "leg_group": np.select([leg == 1, leg == 2, leg == 3, leg >= 4], ["1", "2", "3", "4+"], "unknown"),
        "carrier": f["carrier"],
        "carrier_type": np.where(f["carrier"].isin(REGIONAL), "regional", "mainline"),
        "month": pd.to_datetime(f["forecast_date"]).dt.month.astype(str),
        "direction": f["direction"],
    }, index=f.index)


def slice_table(s: pd.DataFrame, slices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in slices.columns:
        for val, idx in slices.groupby(col).groups.items():
            rows.append({"slice": col, "value": val, **summarize(s.loc[idx])})
    return pd.DataFrame(rows)


def block_bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, metric: str = "brier",
                         block: int = 7, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired block bootstrap over days of mean(metric_a) - mean(metric_b).

    a and b hold per-flight scores for the same flights, with a 'forecast_date' column.
    Negative difference = a is better (lower).
    """
    keys = ["forecast_date", "flight_id"]
    m = a[keys + [metric]].merge(b[keys + [metric]], on=keys, suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("bootstrap needs the same flights in both runs")
    day = m.groupby("forecast_date").agg(sa=(f"{metric}_a", "sum"), sb=(f"{metric}_b", "sum"),
                                         n=(f"{metric}_a", "size")).sort_index()
    sa, sb, n = day["sa"].to_numpy(), day["sb"].to_numpy(), day["n"].to_numpy()
    n_days = len(day)
    n_blocks = int(np.ceil(n_days / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(n_days - block + 1, 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n_days]
    idx = np.minimum(idx, n_days - 1)
    diffs = sa[idx].sum(1) / n[idx].sum(1) - sb[idx].sum(1) / n[idx].sum(1)
    point = sa.sum() / n.sum() - sb.sum() / n.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": metric, "diff": float(point), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(hi < 0 or lo > 0), "n_days": int(n_days)}
