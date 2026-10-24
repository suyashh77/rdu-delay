"""Run one model through the walk-forward harness + leakage tests.

Usage: python scripts/run_experiment.py <model> <run_id> [--fast] [--vs <champion_run_id>]
"""
import argparse
import json
import sys

from rdu.eval import leakage
from rdu.eval.walkforward import ROOT, Config, compare, load_data, run
from rdu.models.baselines import MODELS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("run_id")
    ap.add_argument("--fast", action="store_true", help="every 7th day (screening)")
    ap.add_argument("--vs", help="champion run_id to compare against")
    ap.add_argument("--skip-leakage", action="store_true")
    a = ap.parse_args()

    data = load_data()
    cfg = Config(model=a.model, stride=7 if a.fast else 1)
    make = lambda d: MODELS[a.model]()  # noqa: E731
    summary = run(make, cfg, data, a.run_id)
    out = ROOT / "experiments" / "results" / a.run_id
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
