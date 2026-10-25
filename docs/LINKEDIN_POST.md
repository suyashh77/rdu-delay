# LinkedIn Post: Predicting Flight Delays the Night Before

---

### Part 1: Short LinkedIn Post (Ready to Copy & Paste)

Can machine learning predict flight delays at 11:00 PM the night before?

Two hours before departure, predicting delays is relatively easy. The inbound aircraft is on radar, pilots have filed flight plans, and the FAA has already issued ground stops.

At 11:00 PM the night before, none of that live data exists. 

The airplane scheduled to fly your 4:30 PM departure from Raleigh-Durham (RDU) tomorrow is currently three cities away, about to begin a complex 5-leg journey across the country. Airlines don't post delay warnings at 11:00 PM—most travelers go to bed assuming their flight will leave on time (a 77% historical base rate).

I set out to build an honest, point-in-time probabilistic forecaster for every flight touching RDU the night before. 

Here is what I learned from analyzing 533,000 flights and 1.48 million aircraft legs across the US:

1. **Academic Benchmarks Often Cheat Unintentionally**
Many published papers claim 85–90% accuracy, but a close audit reveals lookahead leakage: using random train/test splits, training on actual realized weather instead of forecasts, or learning from government records written *after* disruptions (like blank aircraft IDs on cancelled flights). When we stripped out those subtle cheats, we forced our model to learn true causality.

2. **Rebuilding Physical Aircraft Chains**
Instead of treating each flight in isolation, we reconstructed the nationwide daily journey of every commercial aircraft in America. Tracking leg sequence, turnaround padding, and scheduled buffers gave our initial LightGBM model a **+7.3% skill boost** over strong Markov baselines.

3. **Forecast Weather Delivers Double-Digit Gains**
Adding 77 point-in-time weather forecast features (TAF, NBM hourly thunderstorm probabilities, and SPC Day 2 storm outlooks frozen at 10:30 PM) produced a **+10.5% gain in delay probability accuracy (Brier score)** and a **+14.0% reduction in expected delay error (CRPS)**.

4. **Where the Signal Actually Lives**
- **Morning departures (6–9 AM):** Weather barely matters. Delays are driven by whether the airplane arrived late the night before and crew rest constraints. (A plane arriving 30+ min late overnight doubles morning delay risk from 8% to 17%).
- **Evening flights (6–11 PM):** Convective summer weather in national hubs cascades down the chain, cutting delay prediction errors by over 1.1 minutes per flight.

5. **Calibration Over Hype**
A forecast is useless if its probabilities aren't trustworthy. Expected Calibration Error dropped by 37% to **0.0093**, and our 80% prediction intervals covered **79.96%** of actual outcomes. When the model predicts a 30% chance of delay, exactly 30 out of 100 flights are delayed.

Full technical breakdown, code, and decision logs below 👇

#MachineLearning #DataScience #Aviation #TimeRelatedForecasting #AI #MLOps #Python

---

### Part 2: Detailed Technical Breakdown & Behind-the-Scenes

For engineers, data scientists, and researchers who want to see under the hood:

```
                                  [ THE ARCHITECTURE ]
                                  
  ┌─────────────────────────────────┐   ┌────────────────────────────────┐   ┌────────────────────────────────┐
  │   1. National Aircraft Chains   │   │   2. Point-in-Time Weather     │   │   3. Baseline Priors           │
  │   • 1.48M legs reconstructed    │   │   • Frozen at 22:30 ET         │   │   • B2: Flight-number history  │
  │   • Leg number of day (1 to 5+) │   │   • TAF wind/vis/gust/tempo    │   │   • B3: Seasonal calendar      │
  │   • Turnaround slack vs typical │   │   • NBM 3h storm probabilities │   │   • B4: Markov delay chain     │
  │   • Block schedule padding      │   │   • SPC Day 2 convective risk  │   │   • Walk-forward out-of-sample │
  └────────────────┬────────────────┘   └───────────────┬────────────────┘   └────────────────┬───────────────┘
                   │                                    │                                     │
                   └────────────────────────────────────┼─────────────────────────────────────┘
                                                        │
                                                        ▼
                                       ┌──────────────────────────────────┐
                                       │    5-Class LightGBM Classifier   │
                                       │    300 trees, lr 0.05, 63 leaves │
                                       └────────────────┬─────────────────┘
                                                        │
                                                        ▼
                                       ┌──────────────────────────────────┐
                                       │    Calibrated Probabilities &    │
                                       │    Continuous Quantiles (0-180m) │
                                       └──────────────────────────────────┘
```

#### 1. Problem Formulation & Point-in-Time Rules
- **Horizon**: Every night at 11:00 PM ET, publish a probabilistic delay forecast for all flights on Day $D$.
- **Information Cutoff**: Only data generated before 23:00 ET on Day $D-1$ is accessible.
- **Evaluation Window**: 2.5-year walk-forward period (2023-01-01 to 2025-06-30). Models are refit walk-forward without random splits.
- **Holdout Period**: 2025-07-01 onward is strictly locked and untouched.

#### 2. The Trap We Audited (DL-021)
In our first machine learning run (`EXP-001`), the model scored an implausible +12.7% CRPS skill gain. An outcome-by-feature audit revealed:
- BTS records flights that *actually flew*, not the original schedule.
- When an airline cancels a flight, they leave `Tail_Number` blank (931 rows on dev, 100% cancelled vs 1.2% base rate).
- When an aircraft breaks down mechanically, the airline swaps planes, creating an abrupt `chain_break` (4,572 rows, 24.3% cancelled vs 1.1% base rate).
- **The Fix**: Both fields were permanently banned. B4 was rebuilt as `B4v2`, and LightGBM trains strictly on known tails while forecasting all flights.

#### 3. Progression of Results Table (Walk-Forward Benchmark: 37,087 Flights)

| Milestone | Model Description | Brier Score (Lower = Better) | CRPS (Minutes Error) | Skill vs Champion | ECE (Calibration Error) | 80% Coverage (PIT) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **B0** | Climatology Base Rate | 0.1618 | 14.07 | — | 0.002 | 80.6% |
| **B1** | Naive Always On-Time | 0.2034 | 16.51 | -25.7% | 0.203 | 50.6% |
| **B2** | Flight Number History | 0.1550 | 13.64 | +4.2% | 0.005 | 80.3% |
| **B3** | Seasonal Profile (Hour × DoW × Mo) | 0.1550 | 13.73 | +4.2% | 0.010 | 80.2% |
| **B4v2** | **Markov Chain on Aircraft Rotations** | **0.1531** | **13.29** | **Champion-000** | 0.0148 | 81.6% |
| **EXP-001b** | **LightGBM (Rotation + Turn Buffers)** | **0.1419** | **12.02** | **+7.33%** | 0.0147 | 79.96% |
| **EXP-002** | **LightGBM + 77 Forecast Weather Features** | **0.1371** | **11.43** | **+10.46%** | **0.0093** | **79.96%** |

#### 4. Statistical Rigor (Block Bootstrap Validation)
To prevent fooling ourselves with small noisy samples:
- Every comparison is evaluated via a **paired 7-day block bootstrap** (2,000 resamples across calendar days).
- **EXP-002 vs EXP-001b (Weather Value-Add)**: Brier diff 95% CI `[-0.00688, -0.00303]`; CRPS diff 95% CI `[-0.8195, -0.4024]`. Both strictly exclude zero.
- **EXP-002 vs B4v2 (Total System Value-Add)**: Brier diff 95% CI `[-0.01833, -0.01406]`; CRPS diff 95% CI `[-2.1132, -1.6670]`.

#### 5. Slice Breakdown Insights
- **Evening Peak**: Hour 18–20 CRPS improved by 1.05 minutes; Hour 21–23 CRPS improved by 1.17 minutes.
- **Downstream Chains**: Leg 4+ CRPS improved from 14.92 to 13.89 minutes.
- **Regional Airlines**: Regional carrier Brier improved from 0.1360 (B4) down to 0.1161 (EXP-002), eliminating the historical penalty contract carriers suffer under simple baselines.
- **Leakage Suite**: Verified 0 late weather products across all 527,415 flights; passed future-scramble tests, canary cheat detection, and label-shuffling tests.
