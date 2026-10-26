# rdu-delay

Code and methods for point-in-time probabilistic flight delay forecasting at Raleigh-Durham International Airport (RDU).


## Operational status (audited October 26, 2026)

The local departures site uses the **B2 flight-number baseline**, not the experimental weather-and-position LightGBM model. Live inputs are timestamped FlightAware schedule snapshots. Training excludes the locked holdout starting July 1, 2025. Weather and full planned aircraft rotations are not yet integrated into live inference.

The existing Windows task `rdu-aeroapi-nightly` runs `python -m rdu.ingest.live pipeline` at 22:40 local Eastern time. It collects tomorrow's departures, publishes a forecast when the schedule is complete, then collects yesterday's actual gate-out times and refreshes outcomes. The $4.50 monthly cap remains enforced. The machine must be awake with the user logged in. This is local publication; no public hosting deployment or Git commit is automated.

Forecast states are explicit: `provisional` before the publication window, `published` for a complete snapshot frozen from 22:30 to strictly before 23:00 Eastern, `reconstructed` when generated after the cutoff, and `unavailable` when no eligible snapshot exists. Only frozen forecasts are scored as live forecasts. Missing actual times stay unknown; scheduled and estimated times are never substituted for gate-out observations.

Immutable forecasts: `data/forecasts/YYYY-MM-DD.json`. Scores: `data/scores/YYYY-MM-DD.json`. Raw API pages and completeness manifests: `data/live/aeroapi/YYYY-MM-DD/`. The browser displays source timestamps and partial-schedule warnings.

Rebuild the site with `.venv\Scripts\python.exe scripts/export_departures_data.py`. Refresh actual departures manually with `.venv\Scripts\python.exe -m rdu.ingest.live outcomes YYYY-MM-DD`, then rebuild. Run tests with `.venv\Scripts\python.exe -m pytest -q`. Recompute the research table from saved matching predictions with `.venv\Scripts\python.exe scripts/update_research_table.py`.

Research status: EXP-003 is a fast screen, and its saved position features contained a last-leg aggregation bug. The corrected cache is versioned. The affected model must be rerun and pass full acceptance before deployment. Its old scores are retained as historical evidence, not validation of the live baseline. See `docs/LIVE_AUDIT_2026-10-26.md` and `docs/LIVE_PIPELINE.md`.

## Overview

This repository contains the model code, feature engineering pipelines, and walk-forward backtesting harness for predicting flight delays using Bureau of Transportation Statistics (BTS) data, point-in-time aviation weather forecasts (TAF, NBM, SPC), and aircraft rotation tracking under a strict 23:00 ET cutoff on day $D-1$.

## Architecture

- `src/rdu/ingest/`: BTS flight data cleaning, schedule views, and timezone-aware cutoff filtering.
- `src/rdu/features/`: Turnaround buffer slack, point-in-time weather feature stores, and overnight aircraft positioning.
- `src/rdu/models/`: Baseline frequency models (B0–B4v2) and LightGBM 5-bucket probability estimators.
- `src/rdu/eval/`: Walk-forward evaluation loop, block bootstrap significance testing, proper scoring rules (Brier, CRPS, log loss, ECE), and automated leakage verification.
- `tests/`: Automated unit and leakage test suite (future-scramble permutation, canary detection, shuffled-label verification).

## Setup & Usage

### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Run Walk-Forward Evaluation
```bash
# Evaluate model across dev period (stride 8 fast screening)
python scripts/run_experiment.py LGBM_FULL EXP-003-fast --fast --retrain-days 28 --vs EXP-002-fast
```

### 4. Interactive Web UIs
- **Simple RDU Departures Forecaster (`web/departures.html`)**: Clean, minimalist white-background operational table showing all RDU departures for the next day, day simulator (e.g. Mon–Sun weekly simulation), 11:00 PM D-1 cutoff predictions vs actual next-day outcomes, interval calibration accuracy, and physical mechanism audit.
- **Benchmark Research Lab (`web/benchmark_lab.html`)**: Interactive model leaderboard, feature importances, reliability curves, and delay probability simulator.

To view locally:
```bash
python -m http.server 8080 --directory web
# Open http://localhost:8080/departures.html
```

## License
MIT
