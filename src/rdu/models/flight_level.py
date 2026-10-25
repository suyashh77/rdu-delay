"""Flight-level LightGBM: 5-bucket classifier, minutes spread by historical within-bucket shape.

Features (EXP-001): rotation + buffer + point-in-time history (baseline forecasts). No weather.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from rdu.eval.metrics import GRID
from rdu.models.baselines import History, bucket_shapes

CATEGORICAL = ["direction", "carrier", "other_airport", "prev_origin"]
NUMERIC = ["rdu_local_hour", "dow", "month", "leg_number", "n_legs_day", "sched_turn_min",
           "crs_elapsed_min", "distance", "crs_dep_hhmm"]
# Never features: chain_break and tail-missing are written after the fact in BTS (DL-021).
PARAMS = {"objective": "multiclass", "num_class": 5, "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 1.0, "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1}
ROUNDS = 300


class LGBMBuckets:
    def __init__(self, store: pd.DataFrame, params: dict | None = None, rounds: int = ROUNDS,
                 wx_store: pd.DataFrame | None = None, pos_store: pd.DataFrame | None = None):
        self.store, self.params, self.rounds = store, {**PARAMS, **(params or {})}, rounds
        self.wx_store, self.pos_store = wx_store, pos_store

    def _frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        d = pd.to_datetime(rows["forecast_date"])
        X = pd.DataFrame({
            "direction": rows["direction"],
            "carrier": rows["carrier"],
            "other_airport": np.where(rows["direction"] == "dep", rows["dest"], rows["origin"]),
            "prev_origin": rows["prev_origin"],
            "rdu_local_hour": rows["rdu_local_hour"],
            "dow": d.dt.day_of_week,
            "month": d.dt.month,
            "leg_number": rows["leg_number"].astype("Float64"),
            "n_legs_day": rows["n_legs_day"],
            "sched_turn_min": rows["sched_turn_min"],
            "crs_elapsed_min": rows["crs_elapsed_min"],
            "distance": rows["distance"],
            "crs_dep_hhmm": rows["crs_dep_hhmm"],
        }, index=rows.index)
        hist_feats = self.store.reindex(rows["flight_id"])
        hist_feats.index = rows.index
        parts = [X, hist_feats]
        if self.pos_store is not None:
            pos_feats = self.pos_store.reindex(rows["flight_id"])
            pos_feats.index = rows.index
            parts.append(pos_feats)
        if self.wx_store is not None:
            wx_cols = [c for c in self.wx_store.columns if c != "wx_issue_max_utc"]
            wx_feats = self.wx_store[wx_cols].reindex(rows["flight_id"])
            wx_feats.index = rows.index
            parts.append(wx_feats)
        X = pd.concat(parts, axis=1)
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype(object), categories=self.cats[c])
        return X.astype({c: float for c in X.columns if c not in CATEGORICAL})



    def fit(self, hist: History) -> None:
        f = hist.flights[hist.flights["bucket"] >= 0]
        f = f[f["flight_id"].isin(self.store.index)]
        # BTS leaves the tail blank on most cancelled flights, so blank chain features would
        # teach "missing = cancelled" (DL-021). Train on known tails; still predict all rows.
        f = f[f["tail"].notna()]
        self.cats = {c: sorted(pd.Series(self._raw_cat(f, c)).dropna().unique()) for c in CATEGORICAL}
        X = self._frame(f)
        self.booster = lgb.train(self.params, lgb.Dataset(X, f["bucket"].to_numpy()), self.rounds)
        self.shape = bucket_shapes(hist.flights)

    @staticmethod
    def _raw_cat(f: pd.DataFrame, c: str):
        if c == "other_airport":
            return np.where(f["direction"] == "dep", f["dest"], f["origin"])
        return f[c].astype(object).to_numpy()

    def predict(self, targets: pd.DataFrame, sched: pd.DataFrame, hist: History) -> np.ndarray:
        missing = ~targets["flight_id"].isin(self.store.index)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} targets have no history features in the store")
        probs = self.booster.predict(self._frame(targets))
        P = np.zeros((len(targets), GRID))
        for d in ["dep", "arr"]:
            m = (targets["direction"] == d).to_numpy()
            P[m] = probs[m] @ self.shape[d]
        return P
