# Start here (prompt for Claude Code)

Paste this into Claude Code from the root of this folder:

> Read `docs/HANDOFF.md` fully, then `docs/DECISION_LOG.md`, `docs/DATA_FREE.md` and `docs/DATA_PAID.md`. Follow the foundations in HANDOFF §2–§5 exactly. Start at HANDOFF §6 (experiment zero): load BTS data with `scripts/fetch_bts.py`, rebuild aircraft chains, build the walk-forward harness, metrics and leakage tests, then baselines B1–B4. After each step, append an entry to `docs/DECISION_LOG.md` using its template (Simple + Technical). Then work the idea backlog in HANDOFF §7 using the experiment protocol in §4: write the experiment card first, keep a change only if it passes the acceptance rule, log every result including failures.

Fetch BTS data (about 20 minutes):

```
python3 -I scripts/fetch_bts.py data/bts 2022-01 2026-08
```
