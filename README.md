# rdu-delay

Code and methods for point-in-time probabilistic flight delay forecasting at Raleigh-Durham International Airport (RDU).

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

## License
MIT
