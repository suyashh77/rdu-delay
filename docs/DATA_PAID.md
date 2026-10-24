# RDU Flight-Delay Forecaster — Paid Data and Services (For Yash)

This file lists everything that costs money or needs a decision from you. Everything else is free and lives in `DATA_FREE.md`.

## Summary

| Item | Needed for | Required? | Cost model | Your decision |
|---|---|---|---|---|
| FlightAware AeroAPI | Tomorrow's schedule, tail numbers, upstream legs, live actuals | **Yes** (or an alternative below) | Personal tier: no monthly minimum, up to $5/month free credit, personal or academic use only. Standard: $100/month minimum | Which tier |
| LLM API (Claude) | News → structured signals | Yes (small) | Per token | Set a monthly cap |
| AeroDataBox (via RapidAPI) | Cheaper alternative to AeroAPI | Optional | Freemium tiers | Test only if AeroAPI is too expensive |
| Google Maps Routes API | Drive time for "when to leave" | **No (v2)** | Per request, monthly free credit | Skip for v1 |

## 1. FlightAware AeroAPI — the one real cost

### Why we need it
- BTS gives perfect history but arrives about 7 weeks late. It has nothing about tomorrow.
- At 11 pm we need: tomorrow's RDU schedule, which aircraft (tail number) flies each flight, and that aircraft's earlier legs (to get leg number and the inbound origin).
- The next day we need actual times to score our forecast. AeroAPI gives both.

### Tiers (from FlightAware's AeroAPI page, checked 2026-10-24)
- **Personal:** no monthly minimum; up to $5/month free usage credit ($10 for ADS-B feeders); limited to personal or academic use; 10 result sets per minute; **no historical data**.
- **Standard:** $100/month minimum; business and B2C use; 5 result sets per second; historical data back to 2011.
- **Premium:** $1,000/month minimum. Not needed.
- Per-query prices are on FlightAware's separate pricing table. Check it before you sign up.

### My recommendation
1. Start on **Personal** during the build and private testing. We do not need AeroAPI history, because BTS covers history for free.
2. Measure real usage for one week: queries per night for RDU scheduled departures and arrivals, plus one tail lookup per aircraft.
3. Before the site goes public, read the AeroAPI terms. A public website may count as B2C use, which needs Standard. Decide then.

### Calls we expect to make each night
- Scheduled departures from KRDU for tomorrow (paged).
- Scheduled arrivals to KRDU for tomorrow (paged).
- For each aircraft with a known registration: its scheduled legs for tomorrow (to find leg number and inbound origin).
- Next morning: actual times for yesterday's flights (for scoring).
- Cost control: cache everything; never re-query a flight already stored; cap total result sets per night in code.

### What you need to do
- Create the AeroAPI account yourself (I will not create accounts or enter payment details).
- Put the API key in a local `.env` file as `AEROAPI_KEY=...`. Never commit it to GitHub.

## 2. LLM API for news signals

- Use: each night, an LLM reads about 20–200 headlines/snippets and outputs structured JSON (event type, airport, start/end time, severity, confidence, source publish time).
- Cost: small at this volume, but set a hard monthly spending cap in the provider console.
- Use the smallest model that passes an accuracy check on 100 hand-labeled headlines.
- What you need to do: create the API key yourself; store it as `ANTHROPIC_API_KEY` in `.env`.

## 3. AeroDataBox (optional fallback)

- Use: airport departure/arrival boards and flight status at lower cost than AeroAPI.
- Risk: tail-number coverage for tomorrow's flights may be weaker. Test on 7 nights and compare against AeroAPI before switching.
- Action: only if AeroAPI costs are too high.

## 4. Not paid, but needs your signup (listed in DATA_FREE.md)

- FAA NOTAM API key (free).
- FAA ASPM account (free; may need approval).
- OpenSky Network account (free; historical access needs a research request). Optional backup for live actual times.
- GitHub account for hosting the site (GitHub Pages) and the nightly job (GitHub Actions free tier).

## 5. Explicitly not buying

- Cirium / OAG schedules: enterprise pricing, not needed.
- Commercial weather APIs: NOAA data is free and better documented.
- TSA wait-time data vendors: TSA is v2, and vendor numbers for RDU disagree with each other.
