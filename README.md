# Medical Drone Network Planner

Course project for **AIDS361 - Case Study of Emerging Areas of Technology**
(B.Tech AI & Data Science, ADGIPS). Topic: **Autonomous Medical Drone Delivery**.
Builds on the synopsis and 12-slide deck already prepared.

## Goal

A Streamlit app that plans and evaluates a drone network for moving blood,
vaccines and samples between 12 hospitals in **Delhi NCR**, and compares it with
road transport. Four modules, one per idea we chose:

| # | Module | What it answers | Files |
|---|--------|-----------------|-------|
| 1 | Drone vs road simulator | How many minutes does a drone save, by hour of day? | `src/simulator.py`, `src/geo.py` |
| 3 | Hub placement | Where should drone bases go for the fastest average delivery? | `src/hubs.py` |
| 4 | Battery / range / payload | How do payload, wind, cold and battery age change range? | `src/battery.py` |
| 5 | Dispatch + priority routing | How does a fleet serve emergency vs routine orders? | `src/dispatch.py` |

Everything is **simulated from stated assumptions**, all in `config.py`. Real
delivery data is not public, so say this clearly in the report.

## Status (update this after every session)

- [x] Scaffold, `config.py`, `data/hospitals.csv` (12 hospitals)
- [x] Module 1: geometry, road/drone time model, no-fly-zone detour
- [x] Module 3: hub placement (optimal exhaustive search + k-means baseline)
- [x] Module 4: energy/range model + ML range predictor (gradient boosting)
- [x] Module 5: event-driven dispatch, Hungarian assignment, priority vs FIFO
- [x] Streamlit app with one tab per module
- [x] 59 automated tests passing (`python -m pytest -q`, about 35 s)
- [ ] Run the app yourself in a browser and click through every tab
- [ ] Verify hospital coordinates on Google Maps
- [ ] Take screenshots, write the report, make the presentation
- [ ] Optional: real road times from OpenStreetMap (`osmnx`), live weather

## Where we left off

**Session 1 (2026-10-16):** all four modules built and tested in the build
sandbox. The app was tested with Streamlit's automated test runner but **not in a
real browser**. **Next step:** run `streamlit run app.py` locally, fix anything
that looks wrong, then move to the report. If you want more features, the
optional items above are the best candidates.

## How to run

```bash
pip install -r requirements.txt
python -m pytest -q          # should print: 59 passed
streamlit run app.py
```

## Key results to quote (from the default settings; re-run to confirm)

- **Module 1:** from AIIMS at 6 pm, average road time about 63 min vs drone about
  10 min across the other 11 hospitals.
- **Module 3:** optimal average delivery time falls as hubs are added:
  8.8 min (1 hub), 7.2 (2), 5.7 (3), 4.4 (4), 3.7 (5). With a 40 km range even one
  hub reaches every hospital within 20 min, so the interesting comparison is
  average time and a stricter target (the app defaults to 10 min). Optimal search
  beats or ties k-means in every case tested.
- **Module 4:** max one-way range 41 km at 3.5 kg and 48 km at 1.5 kg (calibrated
  to the synopsis figure of about 40 km). Headwind, cold, low charge and old
  batteries all reduce it. The ML model reaches R-squared about 0.99, mean error
  about 1 km on held-out simulated flights.
- **Module 5:** when the fleet is big enough, priority and FIFO behave the same.
  When it is overloaded (60 orders in 3 h, 1 drone at each of 3 hubs), priority
  routing gives emergencies about 18 min on average vs about 156 min under FIFO,
  and the drone advantage over road disappears for routine orders because they
  queue. This shows why fleet sizing matters.

## Things to verify before submission

1. **Hospital coordinates are approximate.** Check each on Google Maps and
   correct `data/hospitals.csv`.
2. **All numbers in `config.py` are assumptions** (road speeds by hour, 1.35 road
   circuity, 90 km/h cruise speed, 5 km IGI no-fly circle, battery and power
   figures, priority mix, SLA targets, 6 min turnaround). The no-fly circle is a
   demo value, not the real DGCA airspace rule. Cite them as assumptions.
3. At 6 pm the model gives about 130 min by road to Gurugram from AIIMS, a bit
   above the synopsis range (45-120 min). Lower the evening speeds in
   `ROAD_SPEED_KMH` to match more closely.
4. The ML model is trained on **simulated** flights (physics plus noise), so it
   mainly learns the physics model. Say so; with real logs you would train on those.

## Design notes and limitations

- Dispatch uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`),
  the exact optimum for assigning orders to drones, instead of OR-Tools. It is
  simpler and has no heavy install. Routing is hub-to-hospital and back (no
  multi-stop tours).
- The road baseline assumes an ambulance is always free (no queueing), which
  favours road transport.
- Multiple no-fly zones are handled by adding each zone's detour separately,
  correct only when zones do not overlap.
- Wind is a single constant for the whole flight, applied as headwind outbound
  and tailwind on return in the dispatch tab.
- Hub sites are limited to existing hospitals.

## Resume prompt (paste into a new Claude chat if you run out of tokens)

> I'm continuing my AIDS361 project "Medical Drone Network Planner" (Streamlit,
> Python, Delhi NCR, 12 hospitals). All four modules (drone-vs-road simulator,
> hub placement, battery/range model, dispatch with priority routing) are built
> and tested. I'm attaching the project files and README. Please read the README
> "Status" and "Where we left off", then help me with the next unchecked item,
> keeping assumptions in config.py and keeping the tests passing.

Tips for the free plan: one task per chat, paste only the relevant file and the
exact error message, and update the Status section above at the end of each session.
