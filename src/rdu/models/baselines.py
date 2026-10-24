"""Baselines B1-B4 (HANDOFF 2.7) plus B0 climatology (reference for the shuffled-label test).

Interface every model follows:
    fit(hist)                       hist = History(legs, flights) known at the fit cutoff
    predict(targets, sched, hist)   -> P, shape (len(targets), GRID)
`targets` and `sched` carry schedule columns only; the harness strips outcomes.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdu.eval.metrics import CANCEL, GRID, target_bin
from rdu.ingest.bts import bucket_of

K_POOL = 20  # partial-pooling pseudo-count; F22 tests 5/20/50/200


@dataclass(frozen=True)
class History:
    legs: pd.DataFrame
    flights: pd.DataFrame


def _calendar(f: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(f["forecast_date"])
    return f.assign(month=d.dt.month, dow=d.dt.dayofweek)


def pooled_hist(train: pd.DataFrame, y_bin: np.ndarray, target: pd.DataFrame,
                levels: list[list[str]], k: float = K_POOL) -> np.ndarray:
    """Histogram of y_bin per finest key, shrunk level by level toward coarser keys.

    levels go coarse -> fine. Each level's estimate is (counts + k * parent) / (n + k).
    The coarsest parent is uniform with weight 1, so the first level is ~empirical.
    """
    P = np.full((len(target), GRID), 1.0 / GRID)
    weight = 1.0
    for cols in levels:
        codes, uniq = pd.MultiIndex.from_frame(train[cols]).factorize()
        counts = np.zeros((len(uniq), GRID))
        np.add.at(counts, (codes, y_bin), 1.0)
        idx = uniq.get_indexer(pd.MultiIndex.from_frame(target[cols]))
        c = np.where(idx[:, None] >= 0, counts[np.maximum(idx, 0)], 0.0)
        P = (c + weight * P) / (c.sum(1, keepdims=True) + weight)
        weight = k
    return P


class _FrequencyModel:
    levels: list[list[str]] = []

    def fit(self, hist: History) -> None:
        f = hist.flights[hist.flights["bucket"] >= 0]
        self.train = _calendar(f)
        self.y = target_bin(f["y_capped"], f["bucket"])

    def predict(self, targets: pd.DataFrame, sched: pd.DataFrame, hist: History) -> np.ndarray:
        return pooled_hist(self.train, self.y, _calendar(targets), self.levels)


class B0Climatology(_FrequencyModel):
    levels = [["direction"]]


class B2FlightNumber(_FrequencyModel):
    levels = [["direction"], ["direction", "carrier"], ["direction", "carrier", "month"],
              ["direction", "carrier", "flight_number", "month"]]


class B3Seasonal(_FrequencyModel):
    levels = [["direction"], ["direction", "rdu_local_hour"], ["direction", "rdu_local_hour", "month"],
              ["direction", "dow", "rdu_local_hour", "month"]]


class B1OnTime:
    def fit(self, hist: History) -> None:
        pass

    def predict(self, targets: pd.DataFrame, sched: pd.DataFrame, hist: History) -> np.ndarray:
        P = np.zeros((len(targets), GRID))
        P[:, 0] = 1.0
        return P


def _hour_block(hhmm: pd.Series) -> np.ndarray:
    return np.digitize(hhmm // 100, [8, 11, 14, 17, 20])  # 6 blocks, origin local time


def _slack_group(turn: pd.Series) -> np.ndarray:
    return np.digitize(turn.fillna(-1), [40, 60, 90, 150])  # 5 groups; <40 includes overlaps


def _counts(a: np.ndarray, b: np.ndarray, shape: tuple) -> np.ndarray:
    c = np.zeros(shape)
    np.add.at(c, (*a, b), 1.0)
    return c


class B4Markov:
    """Discrete Markov chain over 5 buckets, propagated down each aircraft's scheduled day.

    leg 1 (or after a chain break / no tail): P(dep | hour block)
    later legs: P(dep | inbound arr bucket, turn-slack group, hour block), pooled to P(dep | inbound arr)
    every leg: P(arr | dep bucket); cancelled dep -> cancelled arr
    Bucket probabilities are spread over minutes with each bucket's historical shape by direction.
    Note: uses the scheduled tail rotation from BTS, i.e. assumes tomorrow's tails are known (see F1).
    """

    def fit(self, hist: History) -> None:
        legs = hist.legs[hist.legs["tail"].notna()]
        dep = bucket_of(legs["dep_delay"], legs["cancelled"], legs["diverted"]).to_numpy()
        arr = bucket_of(legs["arr_delay"], legs["cancelled"], legs["diverted"]).to_numpy()
        ok = (dep >= 0) & (arr >= 0)
        legs, dep, arr = legs[ok], dep[ok], arr[ok]
        hb = _hour_block(legs["crs_dep_hhmm"])
        first = (legs["leg_number"].to_numpy() == 1) | legs["chain_break"].to_numpy()

        self.leg1 = self._norm(_counts([hb[first]], dep[first], (6, 5)) + 1)
        self.t_arr = self._norm(_counts([dep], arr, (5, 5)) + 1)
        self.t_arr[4] = [0, 0, 0, 0, 1]

        # inbound arrival bucket = previous leg's arrival bucket in the same aircraft-day
        prev_arr = pd.Series(arr, index=legs.index).groupby(
            [legs["flight_date"], legs["tail"]]).shift(1).to_numpy()
        later = ~first & ~np.isnan(prev_arr)
        pa = prev_arr[later].astype(int)
        sg = _slack_group(legs["sched_turn_min"])[later]
        parent = self._norm(_counts([pa], dep[later], (5, 5)) + 1)
        fine = _counts([pa, sg, hb[later]], dep[later], (5, 5, 6, 5))
        self.t_dep = (fine + K_POOL * parent[:, None, None, :]) / (fine.sum(-1, keepdims=True) + K_POOL)

        f = hist.flights[hist.flights["bucket"] >= 0]
        y = target_bin(f["y_capped"], f["bucket"])
        self.shape = {}
        for d in ["dep", "arr"]:
            m = (f["direction"] == d).to_numpy()
            s = _counts([f["bucket"].to_numpy()[m]], y[m], (5, GRID))
            self.shape[d] = s / np.maximum(s.sum(1, keepdims=True), 1)
        for d in self.shape:  # a bucket never seen still needs a shape
            self.shape[d][4] = np.eye(GRID)[CANCEL]

    @staticmethod
    def _norm(c: np.ndarray) -> np.ndarray:
        return c / c.sum(-1, keepdims=True)

    def propagate(self, sched: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Dep and arr bucket distributions for every leg in sched (index-aligned)."""
        s = sched.sort_values(["flight_date", "tail", "sched_dep_utc"])
        dep = np.zeros((len(s), 5))
        arr = np.zeros((len(s), 5))
        hb = _hour_block(s["crs_dep_hhmm"])
        sg = _slack_group(s["sched_turn_min"])
        leg = s["leg_number"].fillna(1).to_numpy()
        first = (leg == 1) | s["chain_break"].to_numpy() | s["tail"].isna().to_numpy()
        pos = np.arange(len(s))
        for k in range(1, int(leg.max()) + 1):
            m = leg == k
            f, l = m & first, m & ~first
            dep[f] = self.leg1[hb[f]]
            prev = pos[l] - 1  # sorted, so the previous row is the previous leg
            dep[l] = np.einsum("ni,nij->nj", arr[prev], self.t_dep[:, sg[l], hb[l]].transpose(1, 0, 2))
            arr[m] = dep[m] @ self.t_arr
        order = s.index
        return (pd.DataFrame(dep, index=order).reindex(sched.index).to_numpy(),
                pd.DataFrame(arr, index=order).reindex(sched.index).to_numpy())

    def predict(self, targets: pd.DataFrame, sched: pd.DataFrame, hist: History) -> np.ndarray:
        dep, arr = self.propagate(sched)
        row = sched.index.get_indexer(targets["leg_index"])
        if (row < 0).any():
            raise ValueError("target legs missing from schedule")
        P = np.zeros((len(targets), GRID))
        for d, dist in [("dep", dep), ("arr", arr)]:
            m = (targets["direction"] == d).to_numpy()
            P[m] = dist[row[m]] @ self.shape[d]
        return P


MODELS = {"B0": B0Climatology, "B1": B1OnTime, "B2": B2FlightNumber, "B3": B3Seasonal, "B4": B4Markov}
