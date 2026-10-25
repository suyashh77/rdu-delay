"""Run one model through the walk-forward harness + leakage tests.

Usage: python scripts/run_experiment.py <model> <run_id> [--fast] [--vs <champion_run_id>]
"""
import argparse
import json
import sys

from rdu.eval import leakage
import pandas as pd

from rdu.eval.walkforward import ROOT, Config, compare, finalize, load_data, run
from rdu.models.baselines import MODELS

LEARNED = ["LGBM", "LGBM_WX", "LGBM_FULL"]


def factory(name: str):
    if name == "LGBM":
        from rdu.features import history
        from rdu.models.flight_level import LGBMBuckets
        store = history.load()
        return lambda d: LGBMBuckets(store)
    if name == "LGBM_WX":
        from rdu.features import history, weather
        from rdu.models.flight_level import LGBMBuckets
        store = history.load()
        wx = weather.load()
        return lambda d: LGBMBuckets(store, wx_store=wx)
    if name == "LGBM_FULL":
        from rdu.features import history, position, weather
        from rdu.models.flight_level import LGBMBuckets
        store = history.load()
        wx = weather.load()
        pos = position.load()
        return lambda d: LGBMBuckets(store, wx_store=wx, pos_store=pos)
    return lambda d: MODELS[name]()




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS) + LEARNED)
    ap.add_argument("run_id")
    ap.add_argument("--fast", action="store_true", help="every 8th day (screening; cycles weekdays, DL-019)")
    ap.add_argument("--vs", help="champion run_id to compare against")
    ap.add_argument("--skip-leakage", action="store_true")
    ap.add_argument("--retrain-days", type=int, default=7)
    ap.add_argument("--finalize", action="store_true", help="reuse saved predictions.parquet")
    a = ap.parse_args()

    data = load_data()
    cfg = Config(model=a.model, stride=8 if a.fast else 1, retrain_days=a.retrain_days)
    make = factory(a.model)
    out = ROOT / "experiments" / "results" / a.run_id
    if a.finalize:
        summary = finalize(pd.read_parquet(out / "predictions.parquet"), cfg, data, a.run_id, None)
    else:
        summary = run(make, cfg, data, a.run_id)
    if not a.skip_leakage:
        leak = leakage.run_all(make, data, cfg)
        (out / "leakage.json").write_text(json.dumps(leak, indent=2, default=float))
        summary["leakage_passed"] = leak["passed"]
    if a.vs:
        cmp = compare(a.run_id, a.vs)
        (out / f"compare_vs_{a.vs}.json").write_text(json.dumps(cmp, indent=2, default=float))
        summary["compare"] = cmp
    print(json.dumps(summary, indent=2, default=float))
    return 0 if summary.get("leakage_passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
