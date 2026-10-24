# HANDOFF v2 — RDU Night-Before Flight-Delay Forecaster

Audience: Claude Code (and Yash). Read this whole file before writing code.

Companion files:
- `DECISION_LOG.md` — the running chain of "we thought about this, tested this, chose this, and why." Website-ready. Append to it after every experiment.
- `DATA_FREE.md` — free data sources (agent instructions).
- `DATA_PAID.md` — paid items (Yash decides).
- `scripts/fetch_bts.py` — BTS downloader (already run once: 2022-01 to 2026-08, 56 files, 533,241 RDU flights, 1,483,703 aircraft-chain legs).

---

## 0. How this project works (read first)

This is **not** a fixed build plan. It is an **experiment loop**:

1. Pick an idea from the backlog (§7).
2. Write the hypothesis and the test before running it.
3. Run the test on the walk-forward harness.
4. If it beats the current champion by the acceptance rule (§4), **keep it**. If not, **revert it**.
5. Log the result in `DECISION_LOG.md` either way. Failed ideas are content too.
6. Repeat.

Terminology note: this is iterative experimentation (hill-climbing with guardrails), not reinforcement learning in the ML sense. RL means an agent learns a policy from rewards. We do not need that. Use "experiment loop" on the website.

Only the **foundations** (§2–§5) are fixed. Every modeling choice above the foundations is an experiment, including choices already made in the decision log. Choices marked "decided by reasoning" must be confirmed by an experiment before the final write-up.

## 1. The problem in one paragraph

Every night at 11:00 pm ET, publish a probabilistic delay forecast for every RDU departure and arrival on the next local day. Score yesterday's forecasts automatically. Publish both on a public static site. The hard part: at 11 pm we have none of the short-term information (inbound aircraft's actual departure time, live ATC programs) that makes same-day prediction easy. Prior work shows accuracy drops from about 90% to about 70% without short-term features. Our job is to extract every bit of signal that exists at 11 pm, and to be honest and calibrated about the rest.

## 2. Foundations (fixed; do not change without Yash)

### 2.1 Point-in-time rule
- Every input must carry the time it became available (issue time, publish time, retrieval time).
- A forecast for day D may use only information available before 23:00 ET on day D−1.
- Weather inputs are **forecasts** issued before the cutoff. Observed weather is for labels and diagnostics only.
- Target encodings and rolling stats use only data before the cutoff.
- In backtests, "yesterday's actuals" (BTS for day D−1) count as available at 23:00 on D−1 only for flights that finished before 23:00. Flights still in the air at 23:00 are unknown.

### 2.2 Delay definitions
- Departure delay = actual gate-out − scheduled departure (minutes). Arrival delay = actual gate-in − scheduled arrival.
- On time = delay < 15 min.
- Cancelled and diverted are separate outcomes. Never drop them from evaluation.
- BTS times are local; convert to UTC; handle midnight crossings.

### 2.3 Output per flight
- `p_bucket`: probabilities summing to 1.0 over `on_time (<15)`, `d15_44`, `d45_89`, `d90_plus`, `cancelled`.
- `p_delay15` = sum of the three delay buckets.
- `expected_delay_capped_min`: mean of the predicted distribution, early set to 0, capped at C minutes (C = 180 by default; C is an experiment, M17). Cancellations excluded.
- `typical_if_delayed_min`: capped mean given delay ≥ 15.
- `quantiles`: q05, q10, q25, q50, q75, q90, q95.
- `reasons`: top 3 drivers in plain English.
- Metadata: `model_version`, `champion_id`, `published_at_utc`, `inputs_cutoff_utc`, `tail_known`, `low_confidence_flags`.
- Website headline: "Chance of 15+ min delay: 35%. If delayed, expect about 40 min." Plus bucket bars.

### 2.4 Data splits
- **Train start:** 2022-01-01.
- **Development walk-forward period:** 2023-01-01 to 2025-06-30. All experiments are judged here.
- **Locked holdout:** 2025-07-01 to the latest BTS date. Touch it only at milestones (§4.4), at most 4 times in the whole project. Log every look.
- **Live:** forecasts published after go-live. Live results are the final judge.

### 2.5 Walk-forward harness
- For each forecast day D in the evaluation window: train on all data with dates < D (or the configured window), build features with the point-in-time rule, predict day D, score day D.
- Retraining cadence is configurable (weekly default). Between retrains, only features update.
- **Fast mode** for screening: evaluate every 7th day. **Full mode** for acceptance: every day.
- Output per run: a results folder with per-flight predictions, per-day scores, slice scores, config, git hash and run time.

### 2.6 Metrics
- **Primary (decides keep/revert):**
  1. Brier score on `p_delay15`.
  2. CRPS on the delay distribution (early = 0, capped at 180, cancellations scored as 180 for CRPS).
- **Guardrails (must not get worse):**
  - Expected calibration error (ECE) on `p_delay15`.
  - 80% interval coverage within 75–85%.
  - Multiclass log loss on buckets.
- **Reported (not decision-making):** RMSE on capped mean, MAE on median, AUC, skill score vs each baseline.
- **Slices (reported every run):** hour block, leg number (1, 2, 3, 4+), carrier, mainline vs regional, calm vs storm day (SPC Day 2 ≥ SLGT or NWS warning at RDU or inbound origin), month, departures vs arrivals.

### 2.7 Baselines (build before anything else)
- B1: every flight on time.
- B2: flight-number historical bucket frequencies, same month, point-in-time.
- B3: seasonal profile (day of week × hour block × month) bucket frequencies.
- B4: discrete Markov chain from inbound-delay bucket to departure bucket by turn slack and hour block, propagated from leg 1.
The first champion is the best of B1–B4.

### 2.8 Leakage tests (run automatically every experiment; failure = experiment invalid)
- No feature reads BTS actual times for flights departing on or after the cutoff.
- Weather features come only from products issued before the cutoff.
- Target encodings use only past dates.
- Shuffled-label run scores no better than B1.
- A "future-peek canary": deliberately add tomorrow's actual delay as a feature in a test run; the test must detect and fail it.

## 3. Logging that must start on day one (cannot be recreated later)

- FAA NAS status XML every 15 minutes.
- AeroAPI at 23:00 ET: tomorrow's schedule, tail assignments, and **FlightAware's own estimated times** (for the head-to-head, X1).
- AeroAPI polling during the day: first time each airline posts a delay or a new estimated time (for lead-time-over-airline, X2). Respect the call budget in `DATA_PAID.md`.
- AeroAPI: aircraft end-of-day positions (for overnight-position features, F2).
- Our own forecasts, committed to git before 23:00 ET every night.

## 4. Experiment protocol

### 4.1 Experiment card (write before running)
Create `experiments/EXP-###.md`:
```
ID: EXP-###
Date:
Idea ID (backlog): e.g. F2
Hypothesis: (one sentence, falsifiable)
Change: (exactly what code/config changes)
Champion compared against: (champion_id)
Expected effect: (which metric, which slice, rough size)
Mode: fast screen -> full
Runtime / cost estimate:
```

### 4.2 Acceptance rule (keep the change only if all are true)
1. Primary metrics improve in **full mode**: Brier skill vs champion > 0 **and** CRPS skill vs champion ≥ 0.
2. Improvement is real: a paired block bootstrap over days (blocks of 7 days, 2,000 resamples) gives a 95% interval for the Brier difference that excludes zero.
3. Guardrails hold: ECE does not rise by more than 0.005; 80% coverage stays within 75–85%; log loss does not get worse.
4. No major slice degrades by more than 2% relative Brier (unless the experiment targets a different slice and the trade-off is logged and justified).
5. Leakage tests pass.
6. Complexity check: if the gain is tiny (< 0.5% relative Brier) and the change adds a new data dependency or doubles runtime, revert and log "not worth the cost."

### 4.3 After the run
- Write the result into the experiment card: numbers, slice table, bootstrap interval, decision.
- If kept: bump `champion_id`, tag the git commit `champion-###`.
- Append an entry to `DECISION_LOG.md` (template in that file) with both a **technical** and a **plain-language** explanation.
- Negative results get the same entry. Mark them "Tried, did not help."

### 4.4 Milestones and holdout looks
- Look at the locked holdout only at: (M1) first model that beats all baselines, (M2) after the propagation simulator, (M3) final pre-launch champion, (M4) final write-up.
- If the holdout disagrees with development results, log it and investigate overfitting to the development period before continuing.

### 4.5 Avoiding fooling ourselves
- Count experiments. With many tries, some "wins" are luck. The bootstrap rule plus holdout looks plus live results are the defense.
- Never tune on the holdout or on live data.
- Prefer ideas with a clear mechanism (a reason it should work) over blind feature fishing.
- Re-test the champion against B1–B4 every 10 experiments; if the margin shrinks, something drifted.

## 5. Repository layout

```
rdu-delay/
  README.md
  pyproject.toml              # python 3.11+; lightgbm, catboost, pandas, pyarrow, scikit-learn, mapie,
                              # shap, statsforecast, chronos-forecasting, herbie-data, cfgrib, requests
  .env.example                # AEROAPI_KEY, ANTHROPIC_API_KEY (never commit .env)
  configs/                    # one YAML per experiment and per champion
  data/                       # gitignored; layout in DATA_FREE.md
  experiments/                # EXP-###.md cards + results/ (gitignored large files)
  docs/
    DECISION_LOG.md
    HANDOFF.md
    DATA_FREE.md
    DATA_PAID.md
  src/rdu/
    ingest/                   # one module per source; raw + clean + PROVENANCE line
    features/                 # rotations, buffers, history, weather, calendar, events, news_llm, carrier_health
    models/                   # baselines, day_level, flight_level, propagation, ensemble, calibration
    eval/                     # walkforward, metrics, bootstrap, slices, leakage_tests, reports
    publish/                  # nightly job, scoring, site builder
  scripts/fetch_bts.py
  site/                       # static site; includes an auto-built "Lab Notebook" page from DECISION_LOG.md
  tests/
```

## 6. Starting point (experiment zero)

Build in this order. Each step ends with a logged decision entry.
1. Load BTS; rebuild aircraft daily chains; verify leg numbers by hand on 20 aircraft-days.
2. Walk-forward harness, metrics, bootstrap, slices, leakage tests.
3. Baselines B1–B4 → first champion.
4. First LightGBM hurdle model with rotation + buffer + history features only (no weather) → EXP-001.
Then work the backlog.

## 7. Idea backlog (try, test, keep or drop)

Each idea: **hypothesis** → **how to test** → **what success looks like**. Priority: P1 = do first (high expected value, low cost), P2 = next, P3 = later or speculative. Add new ideas freely; log where they came from.

### 7.1 Features — rotation and propagation
- **F1 (P1) Tail-assignment predictor.** When tomorrow's tail is unknown at 23:00, predict the likely rotation from history (same flight numbers, same carrier, same weekday pattern). Test: accuracy of predicted rotation vs actual on dev period; then model skill with predicted vs true rotations. Success: rotation accuracy reported; skill gap between predicted and true rotation is small.
- **F2 (P1) Overnight aircraft position.** First-wave risk rises if the aircraft ended yesterday late or away from RDU. Test: add `ended_yesterday_late`, `overnight_at_rdu`, `last_arrival_delay_yesterday`. Success: Brier gain on leg-1 flights.
- **F3 (P1) Learned absorption rate** by inbound airport × carrier (× weather class). Hypothesis: airports absorb inherited delay differently (prior work: SEA/ORD/DFW absorb, MCO amplifies). Test: point-in-time absorption rates as features. Success: gain on leg 2+ flights.
- **F4 (P1) Turn and block slack vs typical.** `sched_turn − typical_turn`, `sched_block − typical_block_q50`. Success: gain on afternoon flights.
- **F5 (P2) Cumulative planned slack upstream** (sum of slack across earlier legs today). Success: gain on leg 3+.
- **F6 (P2) Two-hop upstream exposure** (Option C): worst forecast weather at any earlier airport in the planned chain. Success: gain beyond F3/F4 on leg 3+.
- **F7 (P3) Crew proxy.** Same-carrier aircraft swaps and long-sit patterns as weak crew-connection proxies. Speculative.

### 7.2 Features — weather (forecasts only)
- **F8 (P1) TAF at RDU and inbound origin** at scheduled times: wind, gust, visibility, ceiling, thunderstorm/snow/ice codes.
- **F9 (P1) Effective crosswind** from TAF wind vs runway headings (RDU: 5L/23R, 5R/23L, 14/32). Prior practitioner: beat hyperparameter tuning.
- **F10 (P1) NBM 00Z thunderstorm probability by hour** at RDU, inbound origin, top hubs.
- **F11 (P2) SPC Day 2 risk category** at RDU, inbound origin, hubs.
- **F12 (P2) Snow/ice and de-icing flag** (NBM snow/ice + month).
- **F13 (P2) Winds aloft along route** (GFS/NBM upper winds) for block-time deviation on arrivals.
- **F14 (P3) Forecast uncertainty features**: spread between TAF and NBM, or between consecutive NBM runs.

### 7.3 Features — system state and demand
- **F15 (P1) Hub GDP-risk model.** Separate small model: P(ground delay program tomorrow at ATL/CLT/EWR/LGA/ORD/DFW/DCA...) from forecasts (TAF, NBM, SPC) + schedule volume. Use its outputs as features. Research note: forecast-based GDP prediction is an open problem. Success: GDP model AUC reported; flight-level gain on storm days.
- **F16 (P1) Recent carrier health**: rolling 3-day and 7-day on-time and cancellation rate per carrier (system-wide and at RDU), known up to the cutoff. Success: gain during meltdown periods.
- **F17 (P2) Scheduled congestion** per hour at RDU and the inbound origin.
- **F18 (P2) Calendar**: holiday distance, school/university breaks.
- **F19 (P3) Local events** (venue, start/end time).
- **F20 (P3) News LLM flags** (ATC staffing, shutdown, IT outage, strike, runway closure). Kept only if ablation helps.
- **F21 (P3) Aircraft type, mainline vs regional, seats.**

### 7.4 Features — history encodings
- **F22 (P1) Partial pooling strength k** for flight-number / route / carrier encodings. Test k ∈ {5, 20, 50, 200}.
- **F23 (P2) Encoding windows** (30, 90, 365 days; recency-weighted).

### 7.5 Models
- **M1 (P1) Hurdle (P(delay) + conditional quantiles) vs direct 5-class model.**
- **M2 (P2) LightGBM vs CatBoost** (CatBoost won two NASA challenges).
- **M3 (P2) Conditional distribution head**: quantile regression vs Huber vs NGBoost.
- **M4 (P2) Separate departure/arrival models vs one joint model.**
- **M5 (P1) Monte Carlo propagation vs direct per-flight model.**
- **M6 (P1) Shared day-level shock vs independent draws** inside the simulator.
- **M7 (P1) Day-level model**: seasonal-naive vs Lasso vs Chronos-2 (with covariates). Prior work: Lasso beat RF/XGBoost for daily system delay.
- **M8 (P2) Day-level output as a flight-level feature vs as the simulator's shock scale.**
- **M9 (P2) Ensemble/stacking** of LightGBM, Markov B4 and day-level adjustment; weights fit by walk-forward.
- **M10 (P3) Chronos-2 per flight-number series** (challenger).
- **M11 (P3) TabPFN** on a sample (challenger).
- **M12 (P2) Retraining cadence**: weekly vs daily.
- **M13 (P2) Window**: expanding vs sliding 18 months vs recency sample weights.

### 7.6 Calibration and uncertainty
- **M14 (P1) Calibration**: none vs global isotonic vs segmented isotonic (hour block × leg group) vs Platt.
- **M15 (P1) Intervals**: split conformal vs adaptive conformal inference (ACI).
- **M16 (P2) Abstention flags**: thresholds for "low confidence" (tail unknown, missing weather, new route). Success: flagged flights have measurably worse scores; unflagged flights are better calibrated.
- **M17 (P3) Cap value C** for capped mean: 120 / 180 / 240.

### 7.7 Horizons and product
- **P1 (P2) Multi-horizon refreshes**: 23:00, 06:00, and 2 hours before departure. Publish the skill-vs-horizon curve. (Not the headline product; it shows where skill comes from.)
- **P2 (P3) Headline format test**: show 3 formats to 10–20 people; check which one they interpret correctly.

### 7.8 Proof experiments (see §8)
- **X1 (P2)** FlightAware 23:00 estimates vs ours.
- **X2 (P2)** Lead time over first airline delay posting.
- **X3 (P2)** Leakage audit: reproduce a published feature set (e.g., FlightSense) with random split vs our walk-forward; report the gap.
- **X4 (P3)** Generalization: run the champion pipeline unchanged on CLT and ATL.

## 8. Proof plan (how we show we are best)

- **Narrow claim:** "the best-calibrated public night-before forecast for RDU flights, scored on every flight." No broad "best in the world" claims.
- **Pre-registration:** before go-live, commit the metrics, baselines and scoring rules. Link the commit on the site.
- **Public track record:** nightly forecast committed before 23:00 ET; scores computed from public BTS data so anyone can re-score; all misses shown; live reliability diagram and scorecard vs baselines.
- **Head-to-heads:** X1 and X2 after 30 live nights. Check AeroAPI terms before publishing X1.
- **Leakage audit post:** X3.
- **Open benchmark:** after 60–90 nights, release the point-in-time feature dataset + evaluation script + leaderboard (Kaggle or Hugging Face).
- **Generalization:** X4.
- **Write-up:** site "Lab Notebook" (auto-built from `DECISION_LOG.md`), a technical write-up, optional arXiv paper (novel parts: forecast-based hub GDP risk, lead time over airlines, night-before benchmark).
- **One-command reproduction** of the backtest from public data.

## 9. Nightly production job (once a champion beats baselines and AeroAPI is ready)

- 22:30 ET start, publish by 23:00 ET (GitHub Actions or Yash's machine).
- Steps: AeroAPI schedule + rotations + FlightAware estimates → latest TAF/NBM/SPC/NWS → news + LLM extraction (if kept) → day-level model → flight-level model → simulator (if kept) → calibration → `forecasts/YYYY-MM-DD.json` → git commit → site rebuild.
- Next day: fetch actuals, score, append to `scores.csv`, refresh charts.
- Fallbacks: AeroAPI failure → schedule-only forecast with a banner; weather failure → last valid product, flagged.

## 10. Website structure

- **Tonight's forecasts:** search by flight number; bucket bars, headline, top 3 reasons, confidence flag.
- **Scorecard:** live Brier/CRPS vs baselines, reliability diagram, coverage, recent misses.
- **Lab Notebook:** every `DECISION_LOG.md` entry, newest first, each with "Simple" and "Technical" tabs.
- **Method:** architecture diagram, data sources, limits.
- **Pre-registration:** link to the commit.

## 11. Research notes (what others tried and learned)

- GE Flight Quest (Kaggle, Alaska Airlines): arrival-time prediction; the winner was about 40% better than airline estimates; 3rd place chained ridge regressions and gradient boosting.
- NASA Pushback to the Future (2023): all top teams used gradient boosting; per-airport models beat global; deep learning failed out of sample; leakage filtering mattered.
- NASA Run-way Functions (2022): persistence was a strong baseline; lookahead was the main difficulty; logistic regression placed 4th.
- FlightSense (BTS 2018): rotation features gave about 97% of AUC gain; daily origin weather about 3% (caveats: random split, full-data encodings).
- Delay absorption (BTS 2023): absorption score raised AUC 0.865 → 0.898; airports differ; heavy rain flips ATL to amplifying (caveat: possible label leakage).
- Short-term features study: about 70% accuracy without short-term info, about 90% with the aircraft's departure delay.
- Schedule buffers (Brueckner et al. 2024): average flight buffer about 31 min, ground buffer about 38 min (2018); each buffer minute absorbs well under one minute of delay; relation is a max() and nonlinear.
- Distribution shape: spike at zero (about 89% of flights have zero NAS delay), exponential left flank, heavy right tail; two-part logistic + quantile models fit well; long-horizon-only features explain little.
- Probabilistic per-flight (Rotterdam): CRPS about 9–11 min; random-forest intervals too narrow; probabilistic gate assignment cut conflicts by up to 74%.
- Graph networks (STPN): beat LSTM at network level, 6-h horizon; LSTM lost to VAR; propagation effects fade after about 2.5 h.
- LLMs: useful as text readers and explainers; small gains; no validated night-before LLM forecaster.
- Daily NAS bad-day model (UC Berkeley): Lasso beat RF/XGBoost; used realized (not forecast) weather.
- GDP prediction (NASA Ames): logistic regression AUC about 0.79; drivers differ by airport; forecast-based prediction is open.
- Conformal: adaptive methods handle drift better; all struggle in regime shifts.
- Production lessons: leaky features gave fake 85%; effective crosswind beat tuning; inbound delay, flight-number history and turn-vs-typical were top features; calibration by segment; top 3 reasons raised trust; a hub restructure degraded the model in 3 weeks.
- Commercial: FlightCaster (up to 6 h ahead), Google Flights (shows delays only at ≥80% confidence), Flighty (hours ahead, inbound + FAA data), FlightAware Foresight (ETAs; taxi-out from 3 h before pushback). None publicly claims night-before flight-level skill.

## 12. Known limits (state them on the site)

- No crew data; crew and passenger connections spread delay.
- Airline recovery actions are not modeled.
- Same-day surprises (mechanical, medical, security) cannot be forecast the night before.
- Tail assignments may be unknown at 23:00.
- BTS covers reporting carriers; delay-cause codes are airline-reported and noisy.
