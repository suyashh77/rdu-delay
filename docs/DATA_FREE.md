# RDU Flight-Delay Forecaster — Free Data Sources (Agent Instructions)

## 1. Project context (read first)

- Goal: every night at 11:00 pm ET, publish a forecast for every RDU arrival and departure on the next day.
- Each forecast gives: probability of on-time (within 15 min), a delay distribution (quantiles 5/10/25/50/75/90/95), and cancellation probability.
- Training period: **January 2022 to present** (post-COVID only). Do not fetch data before 2022-01-01 unless a source needs a warm-up window.
- Model design (for context): one-hop Markov. An RDU flight's delay depends on the inbound leg's arrival delay plus RDU conditions. Upstream risk is summarized by "leg number of the day" and conditions at the inbound origin airport.

## 2. Hard rules

1. **Point-in-time rule.** For every forecast input, keep the time the information became available (issue time, publish time, retrieval time). The model may only use information available before 11:00 pm ET the night before. A row without an availability timestamp is useless for training.
2. **Forecasts, not observations, are model inputs.** Observed weather is for diagnostics and labels only.
3. **Never fabricate or fill data.** If a source fails, log the failure and move on.
4. **Raw first, then clean.** Save the raw download untouched. Write cleaned files separately.
5. **Provenance log.** For every file, append one line to `data/PROVENANCE.csv`: `source, url_or_endpoint, params, retrieved_at_utc, rows, bytes, notes`.
6. **Be polite.** Throttle requests (default 1 request every 2 s; GDELT 1 every 6 s). Respect robots.txt and terms of use.
7. **Time zones.** Store all timestamps in UTC. Keep the original local time in a separate column when the source gives local time (BTS times are local).

## 3. Folder layout

```
rdu_delay/
  data/
    bts/                 # DONE by Claude (see 4.1)
    weather/taf/         # 4.2
    weather/metar/       # 4.3
    weather/nbm/         # 4.4
    weather/hrrr/        # 4.5 (optional, v2)
    weather/spc/         # 4.6
    weather/warnings/    # 4.7
    weather/nhc/         # 4.8
    faa/advisories/      # 4.9
    faa/nas_status_log/  # 4.10 (live logger, start now)
    faa/notams/          # 4.11
    faa/aspm/            # 4.12
    reference/           # 4.13 – 4.15
    calendars/           # 4.16
    events/              # 4.17
    news/                # 4.18
    PROVENANCE.csv
```

## 4. Sources

Status key: **DONE** = already fetched by Claude. **TODO** = agent fetches. **SIGNUP** = free but needs an account or API key (ask Yash to create it; never create accounts yourself).

### 4.1 BTS Airline On-Time Performance — DONE (Claude)

- What: flight-level records with tail number, scheduled/actual times, wheels-off/on, taxi times, delay minutes by cause, cancellations, diversions.
- Coverage fetched: 2022-01 to 2026-08 (latest BTS release; BTS publishes with roughly a 7-week lag).
- Filter used: every leg flown that day by any aircraft that touched RDU that day (keeps full daily chains for leg-number features).
- Location: `data/bts/bts_rdu_chains_YYYY_MM.parquet`.
- Agent action: **none**, except add each new month when BTS releases it (check `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip`, no zero padding on month). Reuse `scripts/fetch_bts.py`.
- Gotcha: BTS covers reporting carriers only. Regional flights appear under the operating carrier. Delay-cause columns are filled only when arrival delay ≥ 15 min.

### 4.2 Terminal Aerodrome Forecasts (TAF) archive — TODO

- What: official airport weather forecasts, issued about every 6 hours, valid about 24–30 hours. The main point-in-time weather input.
- Endpoint (Iowa Environmental Mesonet, tested and working):
  `https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py?station=KRDU&sts=2022-01-01T00:00Z&ets=2022-02-01T00:00Z&fmt=csv`
- Stations: `KRDU` plus every airport with a nonstop to/from RDU in the BTS data (build the list from `data/bts`: unique `Origin` where `Dest == "RDU"` and unique `Dest` where `Origin == "RDU"`; prefix US airports with `K`). Expect about 40–70 stations.
- Coverage: 2022-01-01 to today. Fetch in monthly chunks per station.
- Keep columns: `station, valid (issue time), fx_valid, fx_valid_end, raw, ftype, is_tempo, is_amendment, sknt, gust, drct, visibility, presentwx, skyc, skyl, product_id`.
- Gotcha: `valid` is the issue time. Use it for the point-in-time rule. Amended TAFs (`is_amendment = True`) issued after 11 pm ET must not be used for that night's forecast.

### 4.3 METAR / ASOS observations — TODO

- What: hourly observed weather. Used for labels, diagnostics and "how wrong was the TAF" features. **Not** a model input for next-day forecasts.
- Endpoint (tested and working):
  `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=RDU&data=all&year1=2022&month1=1&day1=1&year2=2022&month2=2&day2=1&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3`
- Stations: same list as 4.2 (IEM uses the 3-letter code here, without `K`).
- Coverage: 2022-01-01 to today.

### 4.4 NOAA National Blend of Models (NBM) — TODO

- What: hourly gridded forecasts (wind, gusts, visibility, ceiling, precipitation probability, thunderstorm probability, snow, ice).
- Location: AWS Open Data bucket `s3://noaa-nbm-grib2-pds` (public, no credentials; https listing tested and working).
- Run to use: the **00Z run** (available about 9–10 pm EDT / 8–9 pm EST), which is before the 11 pm ET cutoff. Forecast hours covering the next local day (about f004–f030).
- Extract: point values at each station in the 4.2 list. Suggested tool: the `herbie` Python package (`pip install herbie-data`) with `cfgrib`/`eccodes`.
- Variables: 10 m wind speed and gust, visibility, ceiling, 1-h precip probability, thunderstorm probability, snow and ice accumulation, temperature.
- First step: confirm the archive start date for the core product. If it starts after 2022-01, record the actual start in PROVENANCE and stop; do not substitute another model without asking.
- Store as parquet: `station, run_time_utc, valid_time_utc, forecast_hour, variable, value`.

### 4.5 HRRR (optional, v2) — TODO later

- What: high-resolution hourly model; the 00Z run extends to 48 hours.
- Location: `s3://noaa-hrrr-bdp-pds` (public; tested and working).
- Variables: composite reflectivity, CAPE, lightning, visibility, wind gust at stations in 4.2.
- Priority: low. Do this only after 4.2–4.4 are complete.

### 4.6 SPC convective outlooks (Day 1 and Day 2) — TODO

- What: daily thunderstorm risk categories (TSTM, MRGL, SLGT, ENH, MDT, HIGH) as polygons.
- Source: IEM outlook archive `https://mesonet.agron.iastate.edu/request/gis/outlooks.phtml` (shapefile/GeoJSON download), or SPC archive `https://www.spc.noaa.gov/products/outlook/archive/`.
- Use the Day 2 outlook issued before 11 pm ET, plus the Day 1 1200Z/1300Z issuance for diagnostics.
- Output: per station per date, the highest risk category covering that station, with issue time.

### 4.7 NWS watches and warnings archive — TODO

- What: winter storm, ice storm, tropical storm, hurricane, high wind, dense fog watches/warnings.
- Source: IEM `https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml`.
- Output: per station per date, active or forecast products with issue time and valid window.

### 4.8 NHC tropical data — TODO

- What: hurricane forecast tracks and cones.
- Source: `https://www.nhc.noaa.gov/data/` (forecast advisory archive and HURDAT2 best tracks).
- Coverage: Atlantic seasons 2022–present. Keep advisory issue times.

### 4.9 FAA ATCSCC advisory archive (ground stops, ground delay programs) — TODO

- What: historical FAA traffic-management advisories, including ground stops (GS) and ground delay programs (GDP) by airport.
- Source: `https://www.fly.faa.gov/adv/advADB.jsp` (advisory database search).
- Coverage: 2022-01 to today. Filter to RDU, its nonstop airports and the major hubs.
- Keep: advisory number, issue time, type, airport, start/end time, reason, raw text.
- Gotcha: most programs are issued same-day, after the 11 pm cutoff. They are labels and diagnostics, not next-day inputs. Programs announced before 11 pm (rare) are valid inputs.

### 4.10 FAA NAS Status live logger — TODO (start immediately)

- What: current ground stops, GDPs, closures, delays.
- Endpoint (tested and working, XML): `https://nasstatus.faa.gov/api/airport-status-information`
- There is no public archive. Start a logger now that saves the raw XML every 15 minutes with the retrieval timestamp to `faa/nas_status_log/YYYY/MM/DD/HHMM.xml`. Run it on a schedule that survives restarts (cron or a GitHub Action).

### 4.11 FAA NOTAMs — SIGNUP

- What: runway/taxiway closures and other notices known in advance.
- Endpoint: `https://external-api.faa.gov/notamapi/v1/notams` (returns 401 without a key).
- Action: ask Yash to register for a free FAA API key. Then log NOTAMs for RDU daily going forward. Historical NOTAMs are hard to get; do not spend more than one hour searching for an archive.

### 4.12 FAA ASPM — SIGNUP

- What: airport-level performance: called arrival/departure rates, runway configuration, airport capacity.
- Site: `https://aspm.faa.gov/` (account required; access may need FAA approval).
- Action: ask Yash to request access. If approved, pull RDU and hub daily/hourly data 2022–present.

### 4.13 FAA aircraft registry — TODO (run on Yash's computer)

- What: tail number → aircraft type, seats, engine type.
- Download: `https://registry.faa.gov/database/ReleasableAircraft.zip` (blocked from Claude's sandbox; fetch from Yash's machine).
- Output: `reference/faa_registry_MASTER.parquet` and `reference/faa_registry_ACFTREF.parquet`.

### 4.14 BTS T-100 Segment — TODO

- What: monthly seats, passengers and load factor by route and carrier.
- Source: TranStats "T-100 Segment (All Carriers)" download page on `https://www.transtats.bts.gov/`.
- Filter: routes touching RDU, 2022–present.

### 4.15 Airline door-close and boarding cutoffs — TODO (manual table)

- What: minutes before scheduled departure when boarding closes and the door closes, per airline at RDU.
- Source: each airline's official website (contract of carriage or "check-in requirements" page). Record the URL and the date you read it.
- Output: `reference/airline_cutoffs.csv` with `airline, boarding_close_min, door_close_min, checked_bag_cutoff_min, source_url, read_date`.

### 4.16 Calendars — TODO

- Federal holidays: generate with the `holidays` Python package, 2022–2027.
- Wake County Public School System calendars: `https://www.wcpss.net/` (traditional calendar, school years 2021-22 to 2026-27).
- University academic calendars (break start/end, move-in, finals, commencement): NC State registrar, UNC-Chapel Hill registrar, Duke registrar. School years 2021-22 to 2026-27.
- Output: `calendars/calendar_days.csv` with `date, event_type, description, source_url`.

### 4.17 Local events — TODO

- Sports home games 2022–present: NC State (gopack.com), UNC (goheels.com), Duke (goduke.com) football and men's basketball; Carolina Hurricanes (NHL public API `https://api-web.nhle.com/`); Carolina Panthers home games in Charlotte (minor effect, optional).
- Venues: Lenovo Center (formerly PNC Arena), Red Hat Amphitheater, Coastal Credit Union Music Park, Raleigh Convention Center, DPAC. Use official calendars for future events and Wikipedia or archived pages for past events.
- Output: `events/events.csv` with `date, start_local, venue, event, expected_attendance_if_known, source_url, retrieved_at_utc`.

### 4.18 News (for the LLM signals layer) — TODO

- Historical: GDELT DOC 2.0 API `https://api.gdeltproject.org/api/v2/doc/doc` (mode=artlist, format=json). It returned HTTP 429 to Claude, so throttle hard (1 request every 6+ seconds, back off on 429). For bulk history, use the GDELT GKG files or the public BigQuery dataset `gdelt-bq.gdeltv2`.
- Live: Google News RSS (`https://news.google.com/rss/search?q=...`), FAA newsroom, DOT press releases, WRAL, News & Observer RSS.
- Query set: `"Raleigh-Durham" airport`, `RDU airport`, `"ground stop"`, `"air traffic control" staffing`, `"FAA" outage`, `"government shutdown" FAA`, airline name + `outage` (American, Delta, United, Southwest, JetBlue, Spirit, Frontier, Alaska, Allegiant, Breeze), `"tropical storm" "North Carolina"`, `"winter storm" Raleigh`.
- Store: `url, title, source_domain, published_at_utc (GDELT seendate), retrieved_at_utc, query`. Do not store full article text unless the license allows it.
- The LLM extraction step happens later and is not part of this task.

## 5. Out of scope for v1

- TSA wait times and TSA throughput (v2).
- Drive times and traffic (v2 leave-time feature).
- Anything before 2022-01-01.

## 6. Done checklist

- [ ] Station list built from BTS and saved to `reference/stations.csv`
- [ ] 4.2 TAF archive complete for all stations
- [ ] 4.3 METAR archive complete for all stations
- [ ] 4.4 NBM point extracts complete (or archive start date recorded)
- [ ] 4.6–4.8 SPC, warnings, NHC complete
- [ ] 4.9 FAA advisory archive complete
- [ ] 4.10 NAS status logger running on a schedule
- [ ] 4.13–4.17 reference, calendars, events complete
- [ ] 4.18 news history pulled for 2022–present
- [ ] `PROVENANCE.csv` has one line per file
- [ ] Short report: row counts per source, coverage gaps, failures
