"""Point-in-time history features = the baselines' own walk-forward forecasts.

For every RDU flight on day D, store what B2/B3/B4 predicted at 23:00 on D-1. Each value
was built only from history before D's cutoff (the baselines pass the future-scramble
test), so these are safe both as features for day D and as training features for any
later fit. This avoids in-sample target encoding, where a training row's own outcome
leaks into its encoded feature.
"""
from pathlib import Path

import pandas as pd

from rdu.eval.walkforward import ROOT, Config, Data, eval_days, predict_days
from rdu.models.baselines import MODELS

STORE_DIR = ROOT / "data" / "features"
STORE_MODELS = ["B2", "B3", "B4"]
STORE_START, STORE_END = "2022-02-01", "2025-06-30"  # stops before the locked holdout
KEEP = ["p_on_time", "p_d15_44", "p_d45_89", "p_d90_plus", "p_cancelled", "p_delay15",
        "expected_delay_capped_min"]


def build(name: str, data: Data) -> Path:
    cfg = Config(model=name, start=STORE_START, end=STORE_END)
    preds = predict_days(lambda d: MODELS[name](), data, cfg, eval_days(cfg))
    out = preds[["flight_id"] + KEEP].rename(columns={c: f"{name}_{c}" for c in KEEP})
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORE_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    return path


def load() -> pd.DataFrame:
    frames = [pd.read_parquet(STORE_DIR / f"{n}.parquet").set_index("flight_id") for n in STORE_MODELS]
    return pd.concat(frames, axis=1, join="inner")


if __name__ == "__main__":
    import sys
    from rdu.eval.walkforward import load_data
    print(build(sys.argv[1], load_data()))
