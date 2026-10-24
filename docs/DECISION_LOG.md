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

---

## DL-016 — Loading BTS and rebuilding each aircraft's day
Date: 2026-10-24
Status: Tested — kept (verification, not a model experiment)
Question: Can we turn raw BTS records into clean flights with correct times and correct "leg number of the day"?
Options considered: convert local arrival clock times directly (needs midnight and timezone fixes); derive arrival from scheduled departure + scheduled elapsed time.
How we decided: Unit tests (red-eye across midnight and timezones, leg ordering, outcome buckets) and a hand check of 20 random aircraft-days.
Decision: Scheduled departure is converted from the origin's local time to UTC. Scheduled arrival = scheduled departure + scheduled elapsed time. Actual times = scheduled + reported delay. Leg number = order of scheduled departures for one tail number on one BTS flight date. Diverted flights with no arrival time count as "90+ minutes" in the buckets and as 180 minutes for CRPS, like cancellations.

Simple:
Government flight records use each airport's local clock, which makes overnight and cross-country flights easy to get wrong. We avoid that by starting from the departure time and adding the planned flight length. Then we line up every plane's flights for the day in order, so we know whether your flight is the plane's first trip of the day or its fifth. We checked 20 random plane-days by hand; all lined up correctly.

Technical:
2022-01 to 2026-08: 56 monthly files, 1,483,703 legs (every leg of every tail that touched RDU that day), 533,241 RDU rows (266,622 departures, 266,619 arrivals). Outcome mix: on time 77.1%, 15–44 min 10.6%, 45–89 min 5.4%, 90+ 4.9%, cancelled 2.0%; 1,046 diverted. 2,072 legs have no tail number (kept; treated as leg 1 with unknown inbound). Mean 5.2 legs per aircraft-day (max 15). Chain breaks (previous leg ends somewhere other than where this leg starts) on 2.1% of legs, 0.8% excluding legs next to a cancellation; scheduled overlaps on 2.3%. These reflect tail swaps that BTS records imperfectly: in one hand-checked case leg 1 left 8.5 h late yet leg 2 "left" on time, which is impossible for one airframe. So BTS tail rotations are noisy, and propagation features will carry that noise. Data quirks handled: one leg with no flight number (F9 MIA–ATL, 2024-08-25; kept, upstream only); airport code DJT (Palm Beach, from 2026-08) missing from the timezone table, mapped to America/New_York. DST: ambiguous fall-back hour treated as standard time. Outcome availability time = actual gate-in; cancelled = scheduled departure; diverted with no arrival = scheduled arrival + 6 h. Disclosure: while summarizing the data we printed yearly delay and cancellation base rates that include holdout months (no model scores). 2026 cancellation rate is 3.0% vs 1.5–1.6% in 2023–2025.

Linked: HANDOFF §2.1–§2.2, §6 step 1; F1

---

## DL-017 — Interval coverage for an outcome with a big spike at zero
Date: 2026-10-24
Status: Decided by reasoning — foundation change approved by Yash 2026-10-24
Question: How do we check that our 80% ranges really contain the outcome 80% of the time?
Options considered: (a) as written: share of flights whose capped delay falls between the 10th and 90th percentile; (b) randomized PIT coverage: where the outcome falls on the forecast's own probability scale, with partial credit for ties.
How we decided: Reasoning plus a unit test. About 77% of flights have a capped delay of exactly 0 (early or on time). So the 10th percentile is 0 for nearly every forecast, and any honest forecast covers at least 90% under (a). The 75–85% guardrail could never pass, even for a perfect model. With (b), a perfectly calibrated forecast scores exactly 80% (test: 80.0% vs 89%+ for (a)).
Decision: The guardrail uses (b). The naive version (a) is still reported as coverage80_naive.

Simple:
Most flights are "zero minutes late." That pile at zero breaks the usual way of checking a forecast range: even a perfect forecast would look too wide. We switched to a standard fix for lumpy outcomes. It asks where the real outcome landed inside the forecast's own odds, and gives partial credit when it lands on the pile. A perfect forecast scores exactly 80% this way.

Technical:
For outcome y with forecast CDF F, the randomized PIT is U ~ Uniform[F(y−), F(y)]. Coverage credit = |[F(y−), F(y)] ∩ [0.1, 0.9]| / (F(y) − F(y−)), or the indicator F(y) ∈ [0.1, 0.9] when the step has zero width. Under calibration, U ~ Uniform(0, 1), so expected coverage is exactly 0.80. This is the discrete-outcome analogue of central-interval coverage (Czado, Gneiting & Held 2009).

Linked: HANDOFF §2.6, M15

---

## DL-018 — Shuffled-label leakage test: compare against climatology, not "always on time"
Date: 2026-10-24
Status: Decided by reasoning — foundation change approved by Yash 2026-10-24
Question: When we train on scrambled outcomes, what score should count as "suspiciously good"?
Options considered: (a) as written: no better than B1 (every flight on time); (b) no better than B0 climatology (each flight gets the overall historical delay distribution).
How we decided: Reasoning. Any model that just learns the overall delay rate beats B1 on Brier score (about 0.16 vs 0.20). A shuffled-label model still learns that rate, so test (a) passes no matter what and catches nothing. With scrambled labels, the best honest result is climatology. Beating climatology means some real link between inputs and outcomes survived the scramble, i.e. a leak.
Decision: (b), with a tolerance of 0.001 Brier.

Simple:
One safety check scrambles the answers before training. A model trained on scrambled answers should only learn "about 1 in 5 flights is late," nothing more. The written rule compared it to "every flight is on time," which it always beats, so that check could never catch anything. We now compare it to the plain 1-in-5 forecast instead.

Technical:
Brier(B1) = base rate ≈ 0.20; Brier(climatology) = p(1−p) ≈ 0.16. Shuffled runs permute outcome columns jointly across all history rows at every fit. Run on stride-28 days of the evaluation window. Smoke test (2022-03/04): shuffled B4 0.1226 vs climatology 0.1223, pass.

Linked: HANDOFF §2.8

---

## DL-019 — Walk-forward harness, metrics and leakage tests
Date: 2026-10-24
Status: Tested — kept (verification)
Question: Is the test bench itself trustworthy?
Options considered: n/a (foundation, HANDOFF §2.5–§2.8)
How we decided: Unit tests on hand-computed cases plus a deliberate cheat.
Decision: Built as specified, with the changes in DL-017 and DL-018. Every forecast is a probability spread over 0–180 capped minutes plus a cancellation probability. Buckets, the delay probability, the capped mean, the typical delay if delayed and the quantiles are all read from that one distribution.

Simple:
Before testing any model, we tested the test. The scoring math was checked against answers worked out by hand. Then we built a model that cheats by peeking at tomorrow's real delays, and confirmed our checks catch it every time. Honest models pass the same check: scrambling all future data does not change their forecasts at all.

Technical:
For each day D: cutoff = 23:00 ET on D−1. History = legs whose outcome time < cutoff. Targets and tomorrow's schedule are passed with all outcome columns removed. Refit every 7 days. Leakage suite on every run: (1) future-scramble — outcome columns of every leg not known by D's cutoff are permuted, the model is refit and must give identical forecasts on 4 sample days (honest baselines: max change 0.0); (2) canary — a model that puts 50% of its mass on the true outcome must fail test (1) (it does: max change 180); (3) shuffled labels (DL-018); (4) weather issue time — not applicable yet. CRPS is computed exactly on the 1-minute grid with cancellations at 180; checked: point mass gives |k − y|, uniform 0–9 gives 2.85. Block bootstrap: paired, 7-day blocks, 2,000 resamples over days. Slices: hour block, leg group, carrier, mainline vs regional, month, direction. Not yet available: the calm-vs-storm slice (needs SPC/NWS data). Fast mode changed from every 7th day to every 8th day, because a 7-day stride always lands on the same weekday (approved by Yash 2026-10-24). EXP-001's screen was launched before this change and used every 7th day.

Linked: HANDOFF §2.5–§2.8, DL-017, DL-018

---

## DL-020 — First champion: the Markov chain baseline (B4)
Date: 2026-10-24
Status: Tested — kept (champion-000)
Question: Which simple baseline is the bar every later model must beat?
Options considered: B1 every flight on time; B2 flight-number history (same month); B3 seasonal profile (day of week × hour × month); B4 Markov chain passing delay down each aircraft's scheduled day. B0 climatology run as a reference.
How we decided: Full-mode walk-forward on every day 2023-01-01 to 2025-06-30 (295,102 flights, 911 days), weekly refits, all leakage tests passed for every baseline.
Decision: B4 is champion-000.

Simple:
We tried four simple ways to forecast delays. The best one follows each plane through its day: if the plane is likely to arrive late from its previous flight, your flight is more likely to leave late. It beat the others clearly. But it is not best everywhere. On regional airlines and on a plane's later flights of the day, simply looking at how that flight number usually does was better. That tells us what the next model should combine.

Technical:
| Model | Brier | CRPS | Log loss | ECE | Coverage80 (PIT) | Coverage80 (naive) |
|---|---|---|---|---|---|---|
| B0 climatology | 0.1618 | 14.07 | 0.790 | 0.002 | 0.806 | 0.908 |
| B1 always on time | 0.2034 | 16.51 | 3.017 | 0.203 | 0.506 | 0.632 |
| B2 flight number | 0.1550 | 13.64 | 0.770 | 0.005 | 0.803 | — |
| B3 seasonal | 0.1550 | 13.73 | 0.780 | 0.010 | 0.802 | — |
| **B4 Markov** | **0.1539** | **13.46** | **0.758** | 0.016 | 0.821 | 0.932 |

B4 vs B2: Brier skill +0.73%, CRPS skill +1.4%; B4 vs B3: Brier +0.72%, CRPS +2.0%. Paired 7-day block bootstrap (2,000 resamples) intervals exclude zero for Brier and CRPS in both comparisons. Base rate of 15+ minute delay: 20.3%.
Slices (Brier, B4 vs best of B2/B3): better on leg 1 (+0.9%), leg 2 (+1.7%), arrivals (+1.3%), mainline (+2.5%), morning; worse on leg 3 (−0.5%), leg 4+ (−1.2%), evening 18–20 (−1.6%), regional carriers (−9.5%), and the 931 flights with no tail number (−28%; B4 treats them as leg 1).
Caveats: (1) B4 has the worst calibration of the three useful baselines (ECE 0.016 vs 0.005 for B2); calibration is backlog idea M14. (2) B4 uses the real tail rotations from BTS. Live, these must come from tomorrow's schedule. An early AeroAPI check (2026-10-24, 30 flights, about 12 hours ahead) had tail numbers for 28 of 30; F1 measures this properly. (3) B4 transitions ignore carrier, which likely explains the regional gap. (4) B4's full run used code from commit f097685; a later change (one pseudo-observation per bucket shape, commit 0b443f4) changes its forecasts by a negligible amount. Results in experiments/results/B*-full/.

Linked: HANDOFF §2.7, §6 step 3; DL-015; EXP-001; F1, M14

---

## DL-021 — A leak our tests could not see: BTS "schedule" fields written after the fact
Date: 2026-10-24
Status: Tested — dropped (the leaky features); EXP-001 first run invalid
Question: Are all the "schedule" fields in BTS really known the night before?
Options considered: trust the BTS tail rotation as tomorrow's plan; treat fields shaped by same-day operations as outcomes.
How we decided: EXP-001's first screen beat B4 by 7.7% Brier and 12.7% CRPS, far above the 2–5% we expected. We checked the outcomes behind its top features on the dev period.
Decision: Never use `chain_break` or "tail missing" as inputs. B4 no longer restarts a plane's chain at a break. LightGBM trains only on flights with a known tail (flights without one are still forecast and scored).

Simple:
Our first machine-learning model looked too good, so we went looking for a cheat. We found one hiding in the government data. When a flight is cancelled, airlines often leave the plane's ID blank. When a plane breaks down, the airline swaps in another plane, and the records show the swap. Neither of those can be known the night before; they are written down after things go wrong. The model had learned "blank plane ID means cancelled." We removed those clues and are re-running the test.

Technical:
Dev period 2023-01..2025-06. Flights with no tail number: 931, of which 100% were cancelled (vs 1.2% overall). Flights whose previous leg ended elsewhere (chain break, mostly tail swaps): 4,572, of which 24.3% were cancelled and 40.7% delayed 15+ min (vs 1.1% and 20.0% without a break). BTS records the tail that actually flew, so both fields encode same-day disruption. The future-scramble test cannot catch this, because BTS presents these as schedule columns: their values never change when outcomes are permuted. `leg_number` is blank exactly when the tail is blank, so the fix also removes no-tail rows from LightGBM training (they stay in evaluation, never dropped). B4 used breaks to restart a chain, which leaked indirectly through its history features; B4 rerun as B4v2 and its feature store rebuilt. Residual risk: `prev_origin` and `n_legs_day` also come from the realized rotation and can shift after swaps. The clean fix is tomorrow's planned rotation (F1; live AeroAPI tails, 28 of 30 known about 12 h ahead in a first check). Lesson for the protocol: any gain far above expectation gets an outcome-by-feature audit before it counts.

Linked: EXP-001, DL-016, DL-020, F1
