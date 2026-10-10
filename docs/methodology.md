# Methodology

PITWALL AI is an **educational** project. This page explains what it does, which choices were made and,
just as importantly, where it can be wrong.

## 1. Data pipeline

1. **Source.** Historical lap data is requested through the open-source [FastF1](https://github.com/theOehrly/Fast-F1)
   library, which downloads timing data for past sessions and caches it locally (`data/raw/fastf1_cache/`).
2. **Normalisation** (`pitwall_ai/data/cleaning.py`). Each session's laps are mapped onto one fixed
   table: `Season, RoundNumber, EventName, EventId, SessionType, Driver, LapNumber, LapTimeSeconds,
   Compound, TyreLife, Stint, IsPitLap, DataSource`. Lap durations are converted to seconds.
   If FastF1 does not provide a field for a session, it is left empty instead of crashing.
3. **Cleaning.** Rules are applied in this order and the number of rows each rule removes is reported:

   | # | Rule | Why |
   |---|------|-----|
   | 1 | Missing / non-positive lap time | Not a usable measurement |
   | 2 | Lap time outside 40-200 s | Timing artefacts |
   | 3 | Missing driver or lap number | Cannot be identified |
   | 4 | Pit-in and pit-out laps | They include pit-lane time |
   | 5 | Deleted laps; laps flagged inaccurate | Timing system says unreliable |
   | 6 | Laps whose track status is not green (`"1"`) | Yellow flag, Safety Car, VSC, red flag |
   | 7 | Lap 1 | Standing start, congestion |
   | 8 | Compounds other than SOFT / MEDIUM / HARD | Wet-weather running is a different problem |
   | 9 | Missing tyre age | Cannot be used as a feature |
   | 10 | Laps slower than 110 % of the session median | Traffic, mistakes, unreported slow laps |

   Cleaning is a judgement call. Removing slow laps makes the data look "cleaner" than racing really is.
4. **Collection** (`scripts/collect_data.py`). Uses the FastF1 event schedule, skips events in the future,
   skips (and records) events that fail to download, and only writes the CSV if at least one event succeeded.

## 2. Demonstration data

`pitwall_ai/data/demo_data.py` generates **synthetic** laps from a simple formula (circuit pace + driver offset +
compound offset + linear tyre degradation - fuel-burn gain + noise) with a fixed random seed. Circuits and drivers
are invented. Every row has `DataSource = "synthetic"`, and the app labels it everywhere. Results on this data say
whether the *software* works, never how well anything predicts real racing.

## 3. Features and models

* **Target:** lap time in seconds (regression).
* **Inputs known before the lap:** tyre compound, tyre age (`TyreLife`), lap number (a proxy for fuel load),
  driver and event name (circuit identifier).
* **Never used:** the target lap, sector times, future laps, or anything computed from them.
* **Preprocessing:** a scikit-learn `Pipeline` - median imputation for numbers, constant imputation and one-hot
  encoding (unseen categories ignored) for categories.
* **Models:** mean baseline (`DummyRegressor`), Linear Regression, Random Forest (150 trees, `min_samples_leaf=5`).

## 4. Evaluation

Laps from the same race are strongly correlated (same weather, track state, car set-ups). Randomly splitting laps
would put near-copies of test laps in the training set and make results look better than they are.
So the split is **by event** (`GroupShuffleSplit` on `EventId`): no event contributes to both training and test.
`EventId` is only used for grouping, never as a feature.

* Metrics: MAE and RMSE in seconds, and R² (reported as unavailable if it is not meaningful).
* At least **4 distinct events** are required; with fewer the training script stops with an explanation instead
  of silently using a weaker split.
* The "best eligible" model has the lowest held-out MAE **and** beats the baseline. If none does, no model is saved.
* The saved model is trained on the training events only (it is not refit using the test events), so the reported
  metrics describe the saved model.
* **Permutation importance** is computed on the held-out events: shuffle one input and measure how much MAE grows.
  Correlated inputs share importance, so treat it as indicative.
* Only one random split is used; with few events, results can move noticeably with a different seed.

## 5. Strategy simulation

A strategy is a list of stints (compound + number of laps). The simulator computes

```
estimated remaining time = sum(estimated lap times) + sum(pit-stop losses)
```

* Tyre age restarts at 1 after each stop (FastF1's `TyreLife` convention).
* Each stop costs `pit_loss_seconds` exactly once. Zero stops cost nothing.
* Lap times come from an interchangeable *estimator*: (a) **manual assumptions** (editable base pace, compound
  offsets, degradation per lap, fuel gain), (b) an **empirical least-squares fit** of lap time versus tyre age
  per compound from the selected dataset, or (c) the **trained model** evaluated on a lap/age grid.
* The **optimiser** enumerates every legal combination of stop count, compound sequence and stint lengths (subject
  to minimum stint length, optional pit-lap step and the two-compound rule), simulates all of them and sorts by time.
  The first row is, by construction, the lowest estimate under the model.
* **Scenarios** (degradation multiplier, constant pace offset, hypothetical Safety Car) are hypothetical what-ifs.
  There is no training data for weather or Safety Car timing, so these are *not predictions*.

### Monte Carlo

For each of N runs (fixed seed), the simulator draws a multiplier on each compound's degradation (lognormal), an
extra pace offset per compound (normal), and the loss of each pit stop (normal). All strategies share the same
draws, so differences between them are not sampling luck. The reported mean, standard deviation, P10-P90 range and
"fraction of runs in which a strategy was fastest" describe simulated outcomes **under these invented uncertainty
distributions**. They are not statistical confidence intervals for a real race.

## 6. Sources of uncertainty and known limitations

* Lap time depends on traffic, track evolution, weather, fuel load, car set-up, driver management and tyre-set
  history. Most are not inputs.
* Strategies are compared as if the car were alone on track; real strategy is also about track position and rivals.
* Pit-stop loss differs by circuit and by stop (slow stops, traffic in the pit lane); here it is a single number.
* Tyre degradation is modelled as linear in age within a compound, which misses "cliffs".
* The optimiser may choose plans a team would never run (e.g. the same compound twice) because tyre allocation
  and other regulations beyond the two-compound rule are not modelled.
* Absolute lap time at a circuit not seen in training is hard to predict (circuit pace is a major component).
* FastF1 data completeness varies by season and session.

## 7. Reproducing the analysis

```
python scripts/collect_data.py --start-year 2023 --end-year 2024 --max-events 8
python scripts/train_model.py --input data/processed/laps.csv
streamlit run app.py
```

Random seeds are fixed (default 42), so the same data produces the same split and metrics.
