# Milestone M1 — Champion-001 Full Acceptance Run

## Hypothesis & Purpose
Evaluate the full feature combination (F1/F3/F4/F5/F7 rotation & buffers, F8/F10/F11/F12 point-in-time forecast weather, and F2 overnight positioning) against the champion baseline (`B4v2-full`) across all **912 days** (295,102 flights) of the dev period (`2023-01-01` to `2025-06-30`).

If guardrails hold, this crowns `champion-001` per `docs/HANDOFF.md` §4.4.

## Model Configuration
- **Model:** `LGBM_FULL`
- **Features:**
  - Base schedule & rotation: `direction`, `carrier`, `other_airport`, `prev_origin`, `rdu_local_hour`, `dow`, `month`, `leg_number`, `n_legs_day`, `sched_turn_min`, `crs_elapsed_min`, `distance`, `crs_dep_hhmm`
  - Baseline history forecasts: Point-in-time B0, B2, B3, B4 bucket probabilities
  - Weather (`data/features/weather.parquet`): 77 TAF, NBM, and SPC Day 2 convective features issued before 22:30 ET
  - Positioning (`data/features/position.parquet`): 8 overnight aircraft positioning features (yesterday's arrival delay, overnight rest slack, same-airport flag; strictly NaN if landed after 23:00 ET cutoff)
- **Harness:** Walk-forward stride 1 (every day), refit every 28 days, point-in-time cutoff 23:00 ET on day $D-1$.
- **Baseline Comparison:** `B4v2-full` (295,102 flights, Brier `0.15346`, CRPS `13.483 min`).

## Acceptance Guardrails
1. **Brier skill > 0** vs `B4v2-full`
2. **CRPS skill >= 0** vs `B4v2-full`
3. **7-day block bootstrap 95% CI** excludes zero for both Brier and CRPS
4. **Calibration ECE** $\le$ `0.0185` (baseline + 0.005)
5. **80% Central Prediction Interval Coverage** within `[75%, 85%]`
6. **Automated Leakage Suite** passes 100%

## Status
- **Run ID:** `champion-001-full`
- **Started:** 2026-10-25
- **Progress:** Executing in background task
- **Results:** Pending completion
