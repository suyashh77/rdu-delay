# Decision Log — RDU Night-Before Flight-Delay Forecaster

This file is the project's lab notebook. Every choice we make goes here: what we considered, how we tested it, what we chose and why. Each entry has a **Simple** version (for anyone) and a **Technical** version (for practitioners). The website builds its "Lab Notebook" page from this file.

**Status labels**
- **Decided by reasoning** — chosen from logic and prior research. Not yet tested on our data. Must be confirmed or overturned by the linked experiment.
- **Tested — kept** — an experiment showed it helps.
- **Tested — dropped** — an experiment showed it does not help. We keep the entry because failures teach too.
- **Superseded** — replaced by a later decision (link it).

---

## Template (copy for each new entry)

```
## DL-### — <short title>
Date:
Status:
Question:
Options considered:
How we decided: (reasoning / prior research / experiment EXP-### with numbers)
Decision:

Simple:
<2–5 short sentences, plain words, no jargon>

Technical:
<metrics, methods, numbers, caveats>

Linked: <backlog idea IDs, experiment IDs, earlier entries>
```

---

## DL-001 — What exactly are we forecasting?
Date: 2026-10-24
Status: Decided by reasoning
Question: What should the forecast predict, and when?
Options considered: (a) exact delay minutes per flight; (b) delayed yes/no; (c) a probability distribution per flight, published the night before.
How we decided: Prior research and the structure of the problem.
Decision: (c). Every night at 11:00 pm ET, publish a probability distribution of delay for every RDU arrival and departure on the next day.

Simple:
Nobody can know a flight's exact delay the night before. Weather can shift, planes can break, and crews can run late. So we give odds, like a weather forecast does. "35% chance of a delay" is honest. "12 minutes late" pretends to know more than anyone can.

Technical:
Delay outcomes are heavy-tailed with a point mass near zero. A point forecast under squared loss is pulled by rare extreme delays; under absolute loss it collapses to "on time" for most flights. A full predictive distribution supports any downstream decision and can be scored with proper scoring rules (Brier, CRPS, log loss).

Linked: DL-002, DL-004

---

## DL-002 — Why the night before is the hard version of this problem
Date: 2026-10-24
Status: Decided by reasoning (supported by prior research)
Question: Why not predict closer to departure, where accuracy is high?
Options considered: forecast at 11 pm the night before; forecast hours before departure; forecast after the inbound plane departs.
How we decided: Prior research on short-term features, plus the user's real decision timing.
Decision: The product forecast is 11 pm the night before. Later refreshes are optional experiments to show how skill grows with time (idea P1).

Simple:
Once you know when the incoming plane left Atlanta, predicting your delay is fairly easy. But by then you have already planned your day. The useful forecast is the one you get the night before. That forecast is much harder, because the key facts do not exist yet. Our goal is to squeeze every clue that does exist at 11 pm, and to be honest about the rest.

Technical:
A published study found about 70% accuracy without short-term features and about 90% once the aircraft's departure delay is known. Commercial tools (Flighty, Google Flights, FlightCaster) operate hours ahead and rely on inbound-aircraft status and FAA programs. None publicly claims night-before flight-level skill. Even with short-term data, residual uncertainty remains: airborne time varies with winds aloft, holding and reroutes; taxi-out varies; and a departure can be held by a destination ground delay program, a crew issue or a mechanical problem.

Linked: DL-001, backlog P1

---

## DL-003 — Training data: post-COVID only
Date: 2026-10-24
Status: Decided by reasoning
Question: How much history should we train on?
Options considered: 10+ years (2015–present); post-COVID only (2022–present); a recency-weighted mix.
How we decided: Regime change reasoning.
Decision: Train on 2022-01 onward. Test recency weighting later (idea M13).

Simple:
Air travel in 2020–2021 behaved nothing like normal. Airlines also changed routes and schedules a lot after COVID. Old data can teach the model patterns that no longer exist. So we start in January 2022.

Technical:
BTS On-Time data, 2022-01 to 2026-08: 533,241 RDU flights; 1,483,703 legs including every same-day leg of each aircraft that touched RDU (needed for leg-number and rotation features). Pre-2022 data is excluded to avoid structural breaks. Window length and recency weighting are tested in M13.

Linked: M13

---

## DL-004 — Which "average" do we show?
Date: 2026-10-24
Status: Decided by reasoning
Question: Should the headline number be the mean delay, the median delay or something else?
Options considered: raw mean; median; Huber-type robust mean; capped mean; a two-part "chance + typical delay if delayed."
How we decided: Reasoning about heavy tails and readability; prior research using two-part models.
Decision: Headline = "Chance of 15+ min delay: X%. If delayed, expect about Y min." Also store a capped mean (early = 0, cap 180 min) and quantiles.

Simple:
A plain average gets dragged around by rare disasters. One flight that is 5 hours late can make a normal flight look "10 minutes late on average." The middle value (median) is hard for most people to read. So we split the answer into two parts: how likely a delay is, and how long it usually lasts when it happens. People understand that right away.

Technical:
Squared-error-optimal means are dominated by tail events; the median is optimal under absolute loss but uninformative for most flights (usually ≤ 0). A hurdle decomposition, P(delay ≥ 15) × E[delay | delay ≥ 15], matches the empirical spike at zero (one study: about 89% of flights have zero NAS delay) and the heavy right tail (q-exponential). Example: outcomes −5 (75%), +25 (20%), +180 (5%) give raw mean 10.25, capped mean (early = 0, cap 120) 11, and hurdle "25% chance, about 44 min if delayed." Cap value C is tested in M17.

Linked: M1, M3, M17

---

## DL-005 — Prediction intervals, not confidence intervals; show the full distribution
Date: 2026-10-24
Status: Decided by reasoning
Question: How do we show uncertainty?
Options considered: a 95% interval; an 80% interval; probability bars over delay ranges.
How we decided: Reasoning about usefulness and checkability.
Decision: Show probability bars (on time, 15–44, 45–89, 90+, cancelled). Store quantiles; use an 80% prediction interval as the compact summary.

Simple:
A "confidence interval" describes an average over many flights. A "prediction interval" describes one specific future flight. We forecast single flights, so we use prediction intervals. A 95% range is usually so wide it says nothing ("0 to 3 hours late"). Bars that show the chance of each delay range are easier to read and more honest.

Technical:
Confidence intervals quantify parameter uncertainty and shrink with sample size; prediction intervals quantify outcome uncertainty and do not shrink to zero. An 80% interval misses 1 in 5 times, so coverage can be verified within weeks of live data; a 95% interval needs months. Coverage is a guardrail metric (75–85%) and is maintained under drift with adaptive conformal inference (M15).

Linked: M14, M15

---

## DL-006 — How far upstream do we model? (One-hop Markov)
Date: 2026-10-24
Status: Decided by reasoning
Question: A plane flying Seattle → Atlanta → RDU carries Seattle's delay. Do we model every airport in the chain?
Options considered: (A) full chain simulation with weather at every airport; (B) one-hop Markov: model only the inbound leg, summarize upstream risk with leg number and inbound-origin conditions; (C) B plus a cheap "worst weather earlier in the chain" feature.
How we decided: Reasoning about data cost and error compounding.
Decision: Start with B. Test C later (idea F6).

Simple:
Your RDU flight only "feels" earlier problems through one thing: how late the incoming plane lands at RDU. So we focus on that incoming flight. We track how many flights the plane has already flown that day, because delays pile up as the day goes on. A plane on its sixth flight is riskier than one that slept at RDU.

Technical:
The departure delay is approximately Markov in the inbound arrival delay plus local conditions. At 11 pm the inbound arrival delay is unknown, so we model its distribution from inbound-leg features: origin airport, leg number, scheduled time, origin forecast weather, origin day severity and scheduled buffers. This avoids modeling dozens of airports and limits compounding error. Known violations: crew connections and airline recovery actions (swaps, holds).

Linked: F1, F3, F5, F6, M5

---

## DL-007 — Gradient-boosted trees for each flight; foundation models for daily series
Date: 2026-10-24
Status: Decided by reasoning (supported by competition results)
Question: Why not use a time-series foundation model (like Chronos-2) for each flight?
Options considered: LightGBM / CatBoost; time-series foundation models; deep neural networks; tabular foundation models (TabPFN).
How we decided: Data-shape reasoning and prior competition results.
Decision: LightGBM is the flight-level core. Chronos-2 is used where data is a true regular time series (daily airport delay levels). Chronos-2-per-flight and TabPFN are challengers.

Simple:
Time-series foundation models are great at continuing a smooth line, like daily sales. A single flight is not a smooth line. It is a one-off event described by many facts: the route, the time, the weather forecast, the plane's schedule. Tree models handle that kind of data best. We still use the foundation model for the daily airport delay level, which is a smooth line.

Technical:
Per-flight rows are tabular with heterogeneous features; a flight number's daily delay series is sparse, noisy and subject to schedule changes. In NASA's Pushback to the Future challenge, all top solutions used gradient boosting, and deep models failed out of sample. In Run-way Functions, simple models stayed competitive. Challengers M10 (Chronos-2 per flight-number series) and M11 (TabPFN) are run on the same harness.

Linked: M2, M7, M10, M11

---

## DL-008 — Simulate how delays spread through the day
Date: 2026-10-24
Status: Decided by reasoning
Question: How do we turn per-leg risks into a forecast for your specific flight?
Options considered: direct per-flight model only; Monte Carlo simulation over tomorrow's aircraft rotations.
How we decided: Reasoning: at 11 pm the inbound delay must be generated, not looked up.
Decision: Monte Carlo simulator with a shared daily shock per airport. Validated against the direct model (M5, M6).

Simple:
We replay tomorrow a thousand times on the computer. In each replay, we roll the dice for every flight's own problems and pass any lateness down each plane's chain of flights. Bad days hit every flight at once, so the dice are linked within a day. Then we count how often your flight ends up late.

Technical:
Per simulation: draw one day-level shock per airport; draw root delays per leg from the flight-level model; propagate with `dep = max(root_dep, inbound_arr − effective_turn_slack)` and `arr = dep + block_deviation` (learned, partial absorption — one study found each scheduled buffer minute absorbs well under one minute of delay). Cancellation draws remove legs. 1,000 draws yield per-flight empirical distributions. Independence across flights would understate bad-day risk; the shared shock fixes that.

Linked: F3, F4, M5, M6, M8

---

## DL-009 — The LLM reads the news; it does not predict delays
Date: 2026-10-24
Status: Decided by reasoning
Question: What role should a large language model play?
Options considered: LLM predicts delay numbers directly; LLM extracts structured events from news and writes explanations; no LLM.
How we decided: Reasoning about calibration and reproducibility; prior research.
Decision: LLM extracts structured events (type, place, time, severity, source time) and rewrites the top 3 reasons in plain English. It is kept only if ablation shows a gain (idea F20).

Simple:
Language models are good at reading. They are not reliable at guessing numbers. So the AI reads the news ("controllers short-staffed at Newark") and turns it into facts the model can use. The numbers come from the forecasting model, which we can test and check.

Technical:
LLM numeric outputs are poorly calibrated and not reproducible across runs. As feature extractors with a fixed schema and source timestamps, their outputs are auditable and point-in-time safe. Published work found text adds small gains; no validated LLM forecaster exists for night-before horizons.

Linked: F20

---

## DL-010 — Walk-forward testing, an experiment loop and a locked holdout
Date: 2026-10-24
Status: Decided by reasoning
Question: How do we test ideas without fooling ourselves?
Options considered: random train/test split; walk-forward (train on the past, predict the next day, repeat); reinforcement learning.
How we decided: Reasoning; the most common failure in published projects is leakage from random splits.
Decision: Walk-forward evaluation. Every idea runs through an experiment loop: hypothesis → test → keep only if it beats the champion by a statistical rule. A locked holdout (2025-07 onward) is checked only at milestones. Live results are the final judge.

Simple:
We test the model the way it will be used. It learns only from the past and then predicts the next day, over and over. We try one idea at a time. If it clearly helps, we keep it. If not, we drop it and write down why. We also keep a slice of recent data locked away, so we cannot accidentally tune the model to it.

Technical:
Random splits leak future information through shared days, encodings and drift; published projects report inflated scores this way (one practitioner saw 85% accuracy collapse in production). Acceptance: Brier improvement with a 7-day block bootstrap 95% interval excluding zero, CRPS not worse, ECE and coverage guardrails, slice checks and leakage tests. This is iterative experimentation, not reinforcement learning (no agent or reward-driven policy is involved).

Linked: HANDOFF §2.4–§4

---

## DL-011 — Forecast weather only
Date: 2026-10-24
Status: Decided by reasoning
Question: Can we train on the weather that actually happened?
Options considered: observed weather (METAR); forecast weather issued before 11 pm (TAF, NBM, SPC).
How we decided: Leakage reasoning.
Decision: Only forecasts issued before the cutoff are model inputs. Observed weather is used for diagnostics.

Simple:
At 11 pm, nobody knows tomorrow's real weather. We only know the forecast. If we trained on real weather, the model would look brilliant in testing and fail in real life. So we train on the forecasts that existed at the time.

Technical:
Several published delay and GDP models use realized weather and are therefore not forecasts. We use archived TAFs (Iowa Environmental Mesonet, by issue time), NBM 00Z runs (NOAA Open Data on AWS) and SPC Day 2 outlooks. Forecast error is part of the problem we must model.

Linked: F8–F14

---

## DL-012 — TSA wait times are out of version 1
Date: 2026-10-24
Status: Decided by reasoning
Question: Should version 1 include security wait times?
Options considered: include TSA now; flights first, TSA later.
How we decided: Ground-truth availability.
Decision: Flights first. TSA later.

Simple:
TSA line modeling is actually the easier math problem. But there is no reliable public record of real wait times at RDU to check it against. Websites that estimate RDU wait times disagree wildly. Flights have exact, public, official times. We start where we can prove we are right.

Technical:
TSA publishes hourly checkpoint throughput (volume) via its FOIA reading room, not wait times. Third-party RDU averages range from about 7 to about 54 minutes. Volume is a good TSFM target and a queueing-model input, but without wait-time labels it cannot be validated.

Linked: future work

---

## DL-013 — Data sources
Date: 2026-10-24
Status: Decided by reasoning
Question: Where does history and tomorrow's schedule come from?
Options considered: BTS (history), FlightAware AeroAPI, Cirium/OAG, AeroDataBox, scraping airport boards.
How we decided: Coverage, cost and tail-number availability.
Decision: BTS for history (free, flight-level, tail numbers). FlightAware AeroAPI for tomorrow's schedule, tail assignments, live actuals and FlightAware's own estimates.

Simple:
Government records give us years of exact flight times for free. But they arrive about seven weeks late and know nothing about tomorrow. A flight-data service tells us tomorrow's schedule and which plane flies which flight. We pay only for that part.

Technical:
BTS PREZIP monthly files (latest: 2026-08, released 2026-10-19). AeroAPI Personal tier: no monthly minimum, up to $5/month credit, personal/academic only, no historical data; Standard: $100/month minimum. Public-site terms must be checked before launch. Weather, FAA and calendar data are free (see DATA_FREE.md).

Linked: DATA_FREE.md, DATA_PAID.md

---

## DL-014 — What "best" means, and how we will prove it
Date: 2026-10-24
Status: Decided by reasoning
Question: What claim do we make, and what evidence backs it?
Options considered: "best flight delay predictor"; a narrow, testable claim with public evidence.
How we decided: Credibility reasoning.
Decision: Claim "the best-calibrated public night-before forecast for RDU flights, scored on every flight." Back it with pre-registered scoring rules, nightly timestamped forecasts, public re-scoring from BTS, head-to-heads vs FlightAware and airline postings, a leakage audit and an open benchmark.

Simple:
"Best in the world" is impossible to prove, so we do not say it. We make a narrow claim and show all the evidence. We publish our scoring rules before we start. We post every forecast before the flights happen, so we cannot change it later. Anyone can check our scores with free government data, including every miss.

Technical:
Pre-registration via a dated commit; nightly forecasts committed before 23:00 ET; scores reproducible from BTS; live reliability diagram and scorecard vs B1–B4; X1 (FlightAware estimates), X2 (lead time over first airline delay posting), X3 (leakage audit of a published method), X4 (generalization to CLT and ATL); open benchmark after 60–90 nights.

Linked: HANDOFF §8, X1–X4

---

## DL-015 — Start with honest baselines
Date: 2026-10-24
Status: Decided by reasoning
Question: What must the model beat?
Options considered: compare only against other ML models; compare against simple, strong baselines.
How we decided: Prior competitions showed simple baselines are hard to beat.
Decision: Four baselines: everyone on time; flight-number history; seasonal profile; discrete Markov chain. The first champion is the best of these.

Simple:
About three out of four flights are on time anyway. So a "model" that always says "on time" already looks good. We only claim success if we beat that, and three other simple methods, by a clear margin.

Technical:
B1 always on time; B2 flight-number bucket frequencies (same month, point-in-time); B3 day-of-week × hour × month profile; B4 Markov transitions from inbound-delay bucket to departure bucket by turn slack and hour block. In NASA's Run-way Functions challenge, persistence was a strong baseline; accuracy is not used as a metric because of the high on-time base rate.

Linked: HANDOFF §2.7
