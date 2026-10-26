# RDU Night-Before Flight Delay Forecaster

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-17%2F17%20passed-brightgreen.svg)]()
[![Point-in-Time](https://img.shields.io/badge/cutoff-23%3A00%20ET%20(D--1)-indigo.svg)]()
[![Calibration](https://img.shields.io/badge/ECE-0.00786-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

> **Calibrated probabilistic flight delay forecasting for Raleigh-Durham (RDU) at 11:00 PM the night before departure, using point-in-time machine learning, aviation weather forecasts, and aircraft rotation chains.**

---

## The Problem
Two hours before departure, predicting flight delays is relatively easy: inbound aircraft are already in the air, and gate hold announcements are active. 

At **11:00 PM the night before**, none of that live data exists:
- The scheduled aircraft is three cities away, about to fly a complex 4-to-6 leg daily journey tomorrow.
- Crews are resting, and airlines post zero delay warnings.
- Travelers go to bed assuming their flight will leave on time (a 77% historical base rate).
- Many published machine learning studies claim 90% accuracy by inadvertently leaking post-departure data recorded after landing.

This repository provides an **audited, leakage-proof forecasting system** predicting discrete delay probabilities (5 buckets) and continuous delay quantiles (5% to 95%) across a **2.5-year walk-forward backtest** (37,087 flights, 2023–2025).

---

## Key Results (Evaluated Across 2.5 Years)

| Model / System | Horizon | Brier Score (lower is better) | Brier Skill vs Baseline | CRPS Delay Error (minutes) | CRPS Skill vs Baseline | Calibration ECE | 80% Central Interval Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Always On-Time** | 23:00 ET | 0.2034 | -32.8% | 14.28 min | -5.9% | 0.2034 | 0.00% |
| **B0 Climatology** | 23:00 ET | 0.1618 | -5.4% | 13.52 min | -0.3% | 0.0076 | 81.40% |
| **B2 Flight History** | 23:00 ET | 0.1587 | -3.4% | 13.44 min | +0.3% | 0.0112 | 81.65% |
| **B3 Seasonal Calendar** | 23:00 ET | 0.1572 | -2.4% | 13.38 min | +0.8% | 0.0098 | 81.52% |
| **B4v2 Markov Rotation Model** | 23:00 ET | 0.1531 | **0.00%** | 13.29 min | **0.00%** | 0.0148 | 80.95% |
| **EXP-001b (Aircraft Rotations)** | 23:00 ET | 0.1419 | **+7.33%** | 12.02 min | **+9.56%** | 0.0147 | 79.96% |
| **EXP-002 (+ Forecast Weather)** | 23:00 ET | 0.1371 | **+10.46%** | 11.43 min | **+13.99%** | 0.0093 | 79.96% |
| **EXP-003 (+ Overnight Position)** | **23:00 ET** | **0.1366** | **+10.79%** | **11.34 min** | **+14.67%** | **0.00786** | **80.12%** |

* **Probability Calibration:** The 80% central forecast interval captures **80.12%** of actual outcomes. Expected Calibration Error (ECE) fell by 47% down to **0.00786**.
* **Physical Mechanics:** 
  * Convective weather forecasts cut evening departure delay error (Hours 18–23) by **1.17 min per flight**.
  * Overnight aircraft parking location cut early morning departure error (Hours 05–08) by **0.39 min per flight**.

---

## The 5 Data Leaks Caught in Published Literature

When building delay forecasters from Bureau of Transportation Statistics (BTS) data, many academic papers and Kaggle kernels suffer from severe data leakage:

1. **Blank Tail Numbers on Cancellations:** When an airline cancels a flight, they leave `Tail_Number` blank in BTS monthly filings (100% cancellation rate on blank tails). We purged missing-tail indicators from training (DL-021).
2. **Realized Tail Swaps (`chain_break`):** BTS records the physical aircraft that actually flew. Emergency mechanical swaps show up as mismatched origins and destinations, leaking same-day breakdowns. We eliminated swap features.
3. **The 2-Hour Horizon Illusion:** Predicting delays at $T-2$ hours using live taxi-out times and airborne radar is a tactical tracking problem, not a strategic night-before forecast.
4. **Realized Observed Weather (METAR):** Feeding observed departure weather gives the model a crystal ball. We strictly use point-in-time forecasts (TAF, NBM, SPC) issued before 10:30 PM.
5. **Random Train/Test Splits:** `train_test_split(shuffle=True)` leaks nationwide weather across afternoon and morning flights on the same day. We enforced strict walk-forward evaluation.

---

## Interactive Dashboard & Documentation

* **[Interactive Web Dashboard](web/dashboard.html):** Delay probability simulator, live sliders, interactive leaderboard, and STAR story explorer.
* **[Comprehensive Research Write-Up](docs/COMPREHENSIVE_WRITEUP.md):** Full academic whitepaper covering mathematical calibration, baseline derivations, and slice breakdowns.
* **[Lab Decision Log](docs/DECISION_LOG.md):** The lab notebook recording decisions DL-001 through DL-024.
* **[LinkedIn Case Study](docs/LINKEDIN_POST_EXACT.md):** Short summary matching the Google TimesFM 3.0 case study format.

---

## Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/suyashh77/rdu-delay.git
cd rdu-delay
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -e .
```

### 2. Run the Automated Leakage & Unit Test Suite
```bash
pytest -v
```
*(All 17 tests verify zero future data leakage, canary detection, and point-in-time constraints).*

### 3. Run a Walk-Forward Screening Experiment
```bash
# Evaluate the full model across the 2.5-year dev set (fast screen stride 8)
python scripts/run_experiment.py LGBM_FULL EXP-003-fast --fast --retrain-days 28 --vs EXP-002-fast
```

---

## License
MIT License. Bureau of Transportation Statistics (BTS) data is public domain.
