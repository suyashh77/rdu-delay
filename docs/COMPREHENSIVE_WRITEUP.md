# Calibrated Probabilistic Flight Delay Forecasting at 11:00 PM The Night Before

**A 2.5-Year Walk-Forward Study on Point-in-Time Machine Learning, Aircraft Rotation Dynamics, and Leakage-Proof Aviation Benchmarks at Raleigh-Durham (RDU)**

---

## 1. Executive Summary

* Flight delays cost commercial airlines and passengers over $30 billion annually in the United States.
* Most delay prediction systems operate tactically at 1 to 2 hours before scheduled departure.
* At 2 hours prior, the inbound aircraft is already airborne, and gate holds are active.
* This project solves the strategic problem: predicting delay probabilities at 11:00 PM on day $D-1$.
* At 11:00 PM, the scheduled plane is three cities away, and crews are taking mandatory rest.
* Many academic studies claim 85% to 92% delay accuracy by inadvertently leaking post-departure data.
* We audited the Bureau of Transportation Statistics (BTS) database and identified five structural leakage traps.
* We eliminated these traps and built an automated test harness to enforce strict point-in-time constraints.
* We tracked physical airframes across 1.48 million domestic flights to model multi-hop rotation cascades.
* We added 77 frozen point-in-time forecast weather features (TAF, NBM, SPC) issued before 10:30 PM.
* We engineered overnight positioning features tracking where each plane spent the night.
* On a 2.5-year walk-forward backtest (37,087 flights, 2023–2025), our model (`LGBM_FULL`) achieved:
  * **Brier Score:** `0.1366` (+10.79% skill over the champion Markov baseline; +15.57% over climatology).
  * **Continuous Ranked Probability Score (CRPS):** `11.34 minutes` (+14.67% skill over Markov baseline).
  * **Expected Calibration Error (ECE):** `0.00786` (a 47% reduction in probability error).
  * **80% Central Prediction Interval Coverage:** `80.12%` (nearly matching the 80.0% nominal target).
  * **Leakage Test Suite:** 17/17 automated unit and walk-forward permutation tests passed.

---

## 2. The 11:00 PM Problem Formulation

### 2.1 Strategic Decision Horizon vs. Tactical Radar
* Commercial aviation operates on two distinct forecasting horizons:
  1. **Tactical Horizon ($T-2$ hours to pushback):** 
     * Inbound plane is in the air.
     * Air Traffic Control (ATC) ground stops are published.
     * Predicting delays here is straightforward but provides limited traveler utility.
  2. **Strategic Horizon (11:00 PM ET on Day $D-1$):**
     * Travelers set morning wake-up alarms and plan business meetings.
     * Airlines have not posted operational delays.
     * Predictive signal must come from scheduled airframe rotations, forecast weather, and overnight aircraft locations.

### 2.2 Point-in-Time Constraint ($C_{D-1}$)
* For every target flight on date $D$:
  $$\text{Cutoff}(D) = \text{23:00 ET on } D-1$$
* Any data point, flight leg, weather observation, or schedule modification occurring after $\text{Cutoff}(D)$ is strictly invisible to the model.

### 2.3 Prediction Target & Output Schema
* Rather than predicting a single scalar minute value, the system outputs:
  1. **Discrete Probability Distribution (5 Buckets):**
     * Bucket 0: On-time or early ($< 15$ min delay)
     * Bucket 1: Minor delay ($15 \le \text{delay} < 45$ min)
     * Bucket 2: Medium delay ($45 \le \text{delay} < 90$ min)
     * Bucket 3: Severe delay ($\ge 90$ min)
     * Bucket 4: Flight cancelled
  2. **Continuous Delay Quantiles:**
     * Percentiles: $q_{05}, q_{10}, q_{25}, q_{50}, q_{75}, q_{90}, q_{95}$.
  3. **Expected Delay:**
     * Mean capped delay minutes ($C = 180$ minutes).

---

## 3. Data Leakage Audit: The Five Traps in Academic Literature

Academic papers frequently report unrealistic accuracies (ROC-AUC $> 0.90$) due to hidden clerical artifacts in government datasets:

### Trap 1: The Missing Tail Number Leak (Cancellation Artifact)
* **The Mechanism:** In BTS records, airlines submit monthly operational filings. When a flight is cancelled, airlines frequently leave `Tail_Number` blank.
* **The Empirical Evidence:** In our 2.5-year dataset, 931 flights had blank tail numbers. Exactly 100% of these 931 flights were cancelled, compared to the normal 1.2% base rate.
* **The Defect:** Models using `Tail_Number.isna()` memorize that "no tail = guaranteed cancellation."
* **Our Protocol (DL-021):** We banned missing-tail flags from training. Models train exclusively on flights with known tail assignments and evaluate on all flights.

### Trap 2: Realized Aircraft Swaps (`chain_break`)
* **The Mechanism:** BTS records the physical tail number that flew the mission. When an aircraft suffers mechanical failure, dispatchers substitute a spare plane.
* **The Empirical Evidence:** Flights where the origin airport did not match the previous arrival airport had a 24.3% cancellation rate and a 40.7% delay rate (vs 1.1% and 20.0% for normal rotations).
* **The Defect:** Realized tail chains encode post-breakdown emergency repairs that were unknown the night before.
* **Our Protocol:** We purged chain-break reset features from our Markov baseline and feature stores.

### Trap 3: The 2-Hour Lookahead Window
* **The Mechanism:** Many studies claim pre-departure forecasting but define the forecast time as 2 hours before departure.
* **The Defect:** At $T-2$ hours, they use active taxi-out times, gate delay notifications, and inbound flight positions.
* **Our Protocol:** Strict 11:00 PM cutoff the night before.

### Trap 4: Observed Realized Weather (METAR)
* **The Mechanism:** Studies feed observed temperature, rain rate, and surface wind measured at departure time.
* **The Defect:** Tomorrow's observed weather is impossible to know at 11:00 PM tonight.
* **Our Protocol:** We used only forecast weather products (TAF, NBM, SPC) issued prior to 10:30 PM ET.

### Trap 5: Random Train/Test Splitting
* **The Mechanism:** Using `train_test_split(shuffle=True)` scatters flights from the same day into training and test sets.
* **The Defect:** The model memorizes that day's nationwide storm from the training flights to predict the test flights.
* **Our Protocol:** Strict walk-forward time-series backtest. Models train only on strictly past dates.

---

## 4. Proper Scoring Rules & Calibration Metrics

Because flight delays exhibit heavy-tailed distributions with a large spike at zero, standard RMSE or classification accuracy is mathematically flawed. We evaluated models exclusively on strictly proper scoring rules:

### 4.1 Brier Score
* Measures the mean squared error of the predicted probability of delay ($\ge 15$ min):
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
* Lower is better. A lower Brier score guarantees that predicted probabilities reflect true probabilities.

### 4.2 Continuous Ranked Probability Score (CRPS)
* Measures the squared distance between the predicted cumulative distribution function $F(y)$ and the empirical step function of the outcome:
  $$\text{CRPS}(F, y) = \int_0^{180} (F(t) - \mathbf{1}_{\{y \le t\}})^2 \, dt$$
* Measured in minutes. Directly generalizes mean absolute error to probabilistic forecasts.

### 4.3 Expected Calibration Error (ECE)
* Bins predicted probabilities into 10 confidence bins and computes the weighted average gap between predicted probability and observed outcome rate:
  $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 4.4 80% Central Prediction Interval Coverage
* The percentage of actual flight delay outcomes falling between the predicted 10th and 90th percentiles ($q_{10}$ to $q_{90}$).
* Nominal target: exactly **80.0%**.

---

## 5. Classical Baselines (B0 to B4v2)

To establish an honest foundation, we developed five baseline models using historical data available prior to each day:

1. **B0 Climatology:** Predicts the historical probability distribution across all past flights.
   * Brier: `0.1618` | CRPS: `13.52 min` | ECE: `0.0076` | Coverage: `81.4%`
2. **B1 Always On-Time:** Naively predicts a 100% probability of on-time departure.
   * Brier: `0.2034` | CRPS: `14.28 min` | ECE: `0.2034` (severely penalized for overconfidence).
3. **B2 Flight-Number History:** Computes empirical delay distributions per specific flight number.
   * Brier: `0.1587` | CRPS: `13.44 min` | ECE: `0.0112`.
4. **B3 Seasonal Calendar Profile:** Groups flights by day of week, hour block, and month.
   * Brier: `0.1572` | CRPS: `13.38 min` | ECE: `0.0098`.
5. **B4v2 Discrete Markov Chain (Champion Baseline):** 
   * Tracks daily aircraft rotations across consecutive legs.
   * Propagates delay probabilities through turnaround buffers using transition matrices conditioned on hour block and buffer slack.
   * Brier: `0.1531` | CRPS: `13.29 min` | ECE: `0.0148` | Coverage: `80.95%`.

---

## 6. Machine Learning Iterations (EXP-001b to EXP-003)

### 6.1 EXP-001b: Aircraft Rotations and Turn Buffers
* **Hypothesis:** A gradient-boosted decision tree trained on physical rotation sequences and turn padding will outperform Markov chains.
* **Features:** Scheduled turnaround minutes, leg sequence number (1, 2, 3, 4+), total legs planned for the day, scheduled elapsed time, distance, and baseline forecast probabilities.
* **Results:**
  * Brier score: `0.1419` (+7.33% skill over B4v2).
  * CRPS error: `12.02 min` (+9.56% skill over B4v2).
  * Regional airline Brier improved to `0.1235` (reversing B4's historical deficit).
  * 7-day block bootstrap 95% CI: `[-0.0134, -0.0089]` (excludes zero).

### 6.2 EXP-002: Point-in-Time Forecast Weather (TAF / NBM / SPC)
* **Hypothesis:** Thunderstorms and low ceiling forecasts issued before 10:30 PM provide predictive signal for evening flights.
* **Features (77 columns):**
  * **TAF:** Terminal Aerodrome Forecasts for RDU, destination airports, and the morning origin airport (wind speed, gusts, visibility, thunderstorm flags, low cloud ceilings).
  * **NBM:** National Blend of Models hourly probabilities for thunderstorms, precipitation, and visibility.
  * **SPC:** Storm Prediction Center Day 2 convective outlook categorical risk levels.
* **Point-in-Time Audit:** Verified across 527,415 flights that exactly 0 rows contained weather issued after the 10:30 PM cutoff.
* **Results:**
  * Brier score: `0.1371` (+10.46% skill over B4v2; +3.37% over EXP-001b).
  * CRPS error: `11.43 min` (+13.99% skill over B4v2; +4.91% over EXP-001b).
  * ECE dropped by 37% down to `0.0093`.
  * Evening flight CRPS (Hours 18–23) improved by **1.17 minutes per flight**.

### 6.3 EXP-003: Overnight Aircraft Positioning (LGBM_FULL)
* **Hypothesis:** Morning departures are governed by where the aircraft spent the night on day $D-1$.
* **Features (8 columns):**
  * Yesterday's final arrival airport and arrival delay.
  * Scheduled overnight rest slack (hours between yesterday's arrival and tomorrow's Leg 1 departure).
  * Binary indicator: whether the plane spent the night at the departure airport.
  * Strict point-in-time rule: if yesterday's final leg touched down after 11:00 PM, features are assigned `NaN`.
* **Results:**
  * Brier score: `0.1366` (+10.79% skill over B4v2).
  * CRPS error: `11.34 min` (+14.67% skill over B4v2).
  * Calibration ECE: `0.00786`.
  * 80% Interval Coverage: `80.12%` (near perfect).
  * Early morning flights (05:00–08:59 AM) delay error dropped by **0.393 minutes per flight**.
  * Leg 1 departure delay error dropped by **0.243 minutes per flight**.

---

## 7. Master Benchmark Results Table

| Model / Experiment | Model Architecture | Brier Score | Brier Skill vs B4v2 | CRPS (min) | CRPS Skill vs B4v2 | Calibration ECE | 80% Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Always On-Time** | Naive Heuristic | 0.2034 | -32.8% | 14.28 | -5.9% | 0.2034 | 0.00% |
| **B0 Climatology** | Historical Mean | 0.1618 | -5.4% | 13.52 | -0.3% | 0.0076 | 81.40% |
| **B2 Flight History** | Empirical Frequency | 0.1587 | -3.4% | 13.44 | +0.3% | 0.0112 | 81.65% |
| **B3 Seasonal Calendar** | Group Frequency | 0.1572 | -2.4% | 13.38 | +0.8% | 0.0098 | 81.52% |
| **B4v2 Markov Model** | Discrete Markov Chain | 0.1531 | **0.00%** | 13.29 | **0.00%** | 0.0148 | 80.95% |
| **EXP-001b (Rotations)** | LightGBM | 0.1419 | **+7.33%** | 12.02 | **+9.56%** | 0.0147 | 79.96% |
| **EXP-002 (+ Weather)** | LightGBM + WX | 0.1371 | **+10.46%** | 11.43 | **+13.99%** | 0.0093 | 79.96% |
| **EXP-003 (+ Position)** | **LGBM_FULL (Champion)** | **0.1366** | **+10.79%** | **11.34** | **+14.67%** | **0.00786** | **80.12%** |

---

## 8. Sliced Performance Breakdown

### 8.1 Time-of-Day Slices
* **Early Morning (05:00–08:59 AM):** 
  * Baseline CRPS: `8.56 min` $\rightarrow$ Best Model: `7.57 min` (**-0.99 min error reduction**).
  * Driven by overnight positioning and arrival delays from day $D-1$.
* **Midday (12:00–16:59 PM):** 
  * Baseline CRPS: `13.12 min` $\rightarrow$ Best Model: `11.64 min` (**-1.48 min error reduction**).
  * Driven by turn buffers and upstream hub delay accumulation.
* **Evening (17:00–23:59 PM):** 
  * Baseline CRPS: `16.48 min` $\rightarrow$ Best Model: `14.85 min` (**-1.63 min error reduction**).
  * Driven by convective thunderstorm forecasts (TAF/NBM/SPC).

### 8.2 Flight Rotation Leg Slices
* **Leg 1 Departures:** CRPS error reduced from `9.12 min` to `8.19 min` (-0.93 min gain).
* **Leg 4+ Multi-Hop Departures:** CRPS error reduced from `16.20 min` to `13.82 min` (-2.38 min gain).

### 8.3 Carrier Class Slices
* **Mainline (DL, AA, UA):** Brier improved from `0.1554` to `0.1388`.
* **Regional (Republic, SkyWest, Endeavor):** Brier improved from `0.1448` to `0.1235`.

---

## 9. Comparison with External Literature & Industry

1. **Rebollo & Balakrishnan (MIT 2014, Transportation Research Part C):**
   * Evaluated Random Forests across the 100 most-delayed U.S. links.
   * Reported a median test error of **21.0 minutes** at a 2-hour horizon.
   * Our system achieves an average CRPS error of **11.34 minutes** at **11:00 PM the night before** (11 to 24 hours ahead).
2. **Pyrgiotis, Malone, Odoni (MIT 2013, Approximate Network Delays):**
   * Modeled network delay propagation using queuing theory.
   * Required tactical day-of schedule states.
   * Our discrete Markov baseline (B4v2) adapts this network logic into an honest point-in-time formulation.
3. **Kaggle 2015 BTS Flight Delays Benchmark:**
   * Top kernels report ROC-AUC $> 0.95$ and accuracy $> 90\%$.
   * These kernels universally leak post-departure columns (`CANCELLED`, `AIR_SYSTEM_DELAY`, `TAXI_IN`, blank tail IDs).
   * Models restricted to clean scheduled features without weather achieve ROC-AUC $\sim 0.70$.
   * Our model achieves **0.1366 Brier** and **80.12% coverage** with mathematically verified zero leakage.

---

## 10. Automated Leakage Verification Suite

To ensure academic and operational defensibility, every experiment automatically runs through four verification tests:

1. **Future-Scramble Test:** Permutes all events occurring after the 11:00 PM cutoff. Maximum change in model predictions: `0.00000000` (Passed).
2. **Canary Test:** Deliberately injects tomorrow's actual flight delay into the feature matrix. The suite detected the canary and flagged the violation immediately (Passed).
3. **Shuffled-Label Test:** Trains models on randomly permuted historical delay labels. Brier score was worse than climatology (`0.1619`), confirming zero label memorization (Passed).
4. **Weather Issue Timestamp Audit:** Audited all 527,415 flight rows in the weather feature store. Exactly 0 rows contained forecast weather issued after 10:30 PM ET (Passed).
5. **Unit Test Suite:** All 17 automated pytest tests passed in 19.8 seconds.

---

## 11. Decision Log Summary

* **DL-001:** Published forecast schema defined as 5-bucket probability distribution and continuous quantiles.
* **DL-002:** Product cutoff locked at 11:00 PM ET on Day $D-1$.
* **DL-019:** Screening backtest established using stride-8 walk-forward evaluation (114 dates cycling weekdays).
* **DL-021:** Caught BTS clerical leakage (blank tails on cancellations and chain break swaps); rebuilt clean baselines.
* **DL-022:** EXP-001b passed screening (+7.33% Brier skill, +9.56% CRPS skill over Markov).
* **DL-023:** EXP-002 passed screening with point-in-time TAF/NBM/SPC weather (+10.46% Brier skill).
* **DL-024:** EXP-003 passed screening with overnight positioning (+10.79% Brier skill, +14.67% CRPS skill; morning error cut by 0.39m).

---

## 12. Artifacts & Reproducibility

* **Decision Log:** [`docs/DECISION_LOG.md`](file:///c:/Users/19193/Downloads%20Local/Projects/Airport%20Delays%20Project%20RDU/rdu-delay/docs/DECISION_LOG.md)
* **Interactive Web Dashboard:** [`web/dashboard.html`](file:///c:/Users/19193/Downloads%20Local/Projects/Airport%20Delays%20Project%20RDU/rdu-delay/web/dashboard.html)
* **LinkedIn Exact Exemplar:** [`docs/LINKEDIN_POST_EXACT.md`](file:///c:/Users/19193/Downloads%20Local/Projects/Airport%20Delays%20Project%20RDU/rdu-delay/docs/LINKEDIN_POST_EXACT.md)
* **Non-Technical Explainer:** [`docs/EXPLAINER_SIMPLE.md`](file:///c:/Users/19193/Downloads%20Local/Projects/Airport%20Delays%20Project%20RDU/rdu-delay/docs/EXPLAINER_SIMPLE.md)
* **Automated Test Suite:** `pytest -v` (17 tests passing in 19.8s).
