Could machine learning predict flight delays at 11:00 PM the night before, without relying on same-day operational data? Two hours before departure, predicting delays is relatively easy. At 11:00 PM the night before, none of that live data exists: the aircraft is three cities away, about to fly a complex 5-leg journey tomorrow. Most travelers go to bed assuming their flight will leave on time (a 77% historical base rate). I wanted to see if real signal exists the night before, or if academic papers claiming 90% accuracy are just leaking the future.

A delay doesn't happen in isolation. It cascades through an airframe across America: an afternoon delay in Raleigh is often born in an early morning turn in Chicago. By stitching together 1.48 million national flights, we can follow each physical aircraft through its day and capture its scheduled turnaround padding and buffer slack.

𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬: Using Bureau of Transportation Statistics (BTS) records for Raleigh-Durham (RDU), I evaluated models on a 2.5-year walk-forward backtest (2023-01 to 2025-06, evaluating every 8th day across 37,087 flights). For every forecast day, models trained strictly on data completed before 11:00 PM ET the night before.

I started with four baselines: climatology, flight-number history, seasonal calendar profiles, and a discrete Markov chain tracking delays down each plane's daily rotation. Next, I trained a 5-class LightGBM model on rotation order and turn slack. Then I added 77 point-in-time forecast weather features (TAF, NBM hourly probabilities, SPC Day 2 convective outlooks) frozen at 10:30 PM. Finally, I added overnight positioning: where the plane slept last night, whether it arrived before 11:00 PM, and its overnight rest slack. Post-hoc BTS fields (blank tail IDs on cancellations and unscheduled swaps) were audited and removed so the model could not cheat.

𝐑𝐞𝐬𝐮𝐥𝐭𝐬 (𝘉𝘳𝘪𝘦𝘳 = 𝘱𝘳𝘰𝘣𝘢𝘣𝘪𝘭𝘪𝘵𝘺 𝘦𝘳𝘳𝘰𝘳 𝘴𝘲𝘶𝘢𝘳𝘦𝘥, 𝘭𝘰𝘸𝘦𝘳 𝘪𝘴 𝘣𝘦𝘵𝘵𝘦𝘳; 𝘊𝘙𝘗𝘖 = 𝘥𝘦𝘭𝘢𝘺 𝘮𝘪𝘯𝘶𝘵𝘦𝘴 𝘦𝘳𝘳𝘰𝘳, 𝘭𝘰𝘸𝘦𝘳 𝘪𝘴 𝘣𝘦𝘵𝘵𝘦𝘳):
Baselines: B0 Climatology 0.1618 Brier, Markov chain 0.1531 Brier / 13.29 min CRPS
Rotation & buffers (EXP-001b): 0.1419 Brier, 12.02 min CRPS (+7.33% skill)
+ Point-in-time weather (EXP-002): 0.1371 Brier, 11.43 min CRPS (+10.46% skill vs Markov)
+ Overnight positioning (EXP-003): 0.1366 Brier, 11.34 min CRPS (+10.79% skill vs Markov)

Weather and overnight features split the day cleanly:
Weather drove evening flights (Hours 18–23), cutting delay error by 1.17 minutes per flight as hub thunderstorms rippled into RDU. Overnight positioning drove morning departures (Hours 05–08): an aircraft arriving 30+ min late overnight doubled morning delay risk from 8% to 17%, and ending yesterday at another airport tripled it to 32%. Feeding this cut morning CRPS error by 0.39 minutes.
Uncertainty: An 80% range should contain about 80% of outcomes. The final model covered 80.12%. Expected Calibration Error (ECE) dropped by 47% from 0.0148 down to 0.00786.

𝐓𝐚𝐤𝐞𝐚𝐰𝐚𝐲: Real predictive signal exists the night before, but it lives in physical constraints: where the plane spent the night and afternoon convective weather in upstream hubs. Honest point-in-time features deliver double-digit gains over classical aviation baselines, without needing live radar or lookahead cheats.

Would be interesting to see if foundation models (like TimesFM or Chronos) can match tree models on flight-level tabular rotations, or if network-level graph simulators are needed to model multi-hop airline recovery?

Full write-up: [https://your-site.com/explainer]
Code and methods: [https://github.com/your-username/rdu-delay]

#MachineLearning #DataScience #Aviation #TimeSeries #Forecasting #MLOps
