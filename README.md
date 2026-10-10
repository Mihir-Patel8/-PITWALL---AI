# PITWALL AI — Formula 1 Race Strategy Intelligence System

> An independent, educational analytics project. **Not affiliated with or endorsed by Formula One, the FIA, any team or driver.**

## 1. Summary

PITWALL AI is a small, integrated Python application that combines **historical Formula 1 lap data**, **machine-learning
lap-time models** and a **pit-stop strategy simulator** behind an interactive Streamlit dashboard. It contains a data
pipeline, an event-level machine-learning evaluation, a testable simulation engine, a UI and an automated test suite.

## 2. Problem statement

*How does lap time change as tyres age? Can history predict lap times at races the model has never seen? How would an
earlier or later pit stop change the estimated time to the flag?* PITWALL AI explores these questions with transparent
methods and states clearly where the answers are uncertain.

## 3. Main features

| Area | What it does |
|------|--------------|
| **Overview** | Data mode (DEMO / HISTORICAL), record counts, lap-time and tyre-age charts |
| **Race Explorer** | Loads a real session through FastF1 (season, Grand Prix, session, driver): lap times, tyre ages, stints, summary tables |
| **Strategy Lab** | Compares one-, two- (and more) stop strategies; ranks by estimated remaining time; hypothetical scenarios; Monte Carlo spread |
| **Model Performance** | Shows *only genuine saved* evaluation results: baseline vs Linear Regression vs Random Forest, residuals, permutation importance |
| **Methodology** | The cleaning rules, evaluation design, simulator assumptions and limitations |
| **Demo mode** | A deterministic, clearly labelled **synthetic** dataset so everything runs offline |

## 4. Technology stack

Python 3.12 · FastF1 · pandas · NumPy · scikit-learn · Plotly · Streamlit · Joblib · pytest · Ruff · GitHub Actions.
(SQLite, Docker, FastAPI, XGBoost and SHAP are deliberately *not* used; see Future improvements.)

## 5. System architecture

```
 FastF1 (online, cached)                     synthetic generator (offline)
        │                                              │
        ▼                                              ▼
 data/fastf1_loader ──► data/cleaning ◄── data/demo_data
        │                    │
        ▼                    ▼
 data/dataset_builder   clean lap table (CSV / DataFrame)
 (scripts/collect_data)       │
                              ├──► ml/features → ml/training (event-grouped split) → ml/evaluation
                              │        (scripts/train_model)         │
                              │                                      ▼
                              │                       artifacts/ model.joblib + evaluation.json
                              ▼                                      │
              strategy/estimators ◄──────────────────────────────────┘
                     │  (manual | empirical | trained model)
                     ▼
         strategy/simulator ── scenarios ── optimizer
                     │
                     ▼
      charts.py ──► views/* ──► app.py (Streamlit)
```

The strategy package never imports Streamlit and never needs to know which estimator produced the lap times.

## 6. Repository structure

```
PITWALL-AI/
├── app.py                      Streamlit entry point
├── pitwall_ai/
│   ├── config.py               paths, constants, schema, colours
│   ├── charts.py               Plotly figures (empty-safe)
│   ├── data/                   fastf1_loader, cleaning, demo_data, dataset_builder
│   ├── ml/                     features, training, evaluation, prediction
│   ├── strategy/               estimators, simulator, optimizer, scenarios
│   └── views/                  one module per dashboard section
├── scripts/                    collect_data.py, train_model.py
├── tests/                      offline pytest suite
├── docs/methodology.md
├── data/{raw,processed}/       (generated data, git-ignored)
├── artifacts/                  (trained model + evaluation, git-ignored)
└── .github/workflows/tests.yml
```

## 7. Installation

**Prerequisites:** Python 3.12, Git, and (recommended) Visual Studio Code.

### Windows (PowerShell, VS Code)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
In VS Code choose **Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`**.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

What the commands do: the first creates an isolated environment (`.venv`); activation makes `python`/`pip` use it;
`pip install -r requirements-dev.txt` installs the runtime and development dependencies; `pytest` runs the tests;
`streamlit run app.py` starts the dashboard at <http://localhost:8501>. **The app works immediately in demo mode.**

### Troubleshooting

| Problem | Try this |
|---------|----------|
| `ModuleNotFoundError` | Your environment is probably not active (prompt should show `(.venv)`). Activate it and re-run `python -m pip install -r requirements-dev.txt`. Run commands from the repository root. |
| Wrong interpreter in VS Code | Select the `.venv` interpreter (see above) and open a **new** terminal. |
| `streamlit` not found | Use `python -m streamlit run app.py`. |
| Dependency install fails | Upgrade pip first; confirm `python --version` is 3.12.x; read the first error line (often a version or network issue) before changing anything else. |
| FastF1 load fails | Check your connection and retry (the first download of a session needs internet). Very recent or future sessions may not be published. Corrupt cache: delete `data/raw/fastf1_cache/`. |
| "No trained model" | Expected until you run the training command below. The simulator and demo mode work without it. |

## 8. Data collection (needs internet the first time)

```powershell
python scripts/collect_data.py --start-year 2023 --end-year 2024 --max-events 4
```

Arguments: `--start-year`, `--end-year` (required, ≥ 2018), `--max-events N` (cap on total events, handy for testing),
`--sessions R S ...` (default `R`), `--output PATH` (default `data/processed/laps.csv`).

The script reads the FastF1 event schedule, skips future events, downloads and cleans each session, **skips and
records any event that fails** (it never corrupts the good ones), and writes `data/processed/laps.csv` plus
`collection_report.json` only if at least one event succeeded. The first download may take several minutes; later
runs reuse the FastF1 cache in `data/raw/fastf1_cache/`. To rebuild from scratch delete `data/raw/fastf1_cache/` and/or
`data/processed/*`.

## 9. Model training

```powershell
python scripts/train_model.py --input data/processed/laps.csv
```

Options: `--output-dir` (default `artifacts/`), `--test-size` (fraction of **events** held out, default 0.25),
`--seed` (default 42), `--allow-synthetic` (software testing only).

It validates the file, trains a mean baseline, Linear Regression and a Random Forest, evaluates them on **held-out
events**, prints a summary, and saves `artifacts/lap_time_model.joblib`, `artifacts/evaluation.json` and
`artifacts/test_predictions.csv`.

**Why several events are needed:** evaluation needs events the model has never seen, so at least **4 distinct events**
are required. With fewer, training stops with an explanation rather than substituting a weaker split.
If no real model beats the baseline, no model is saved and the report says so (exit code 2).
Training on synthetic data is refused unless you pass `--allow-synthetic`.

## 10. How the application works

* The sidebar chooses the data source (demo, or your collected dataset when it exists) and the section.
* **Race Explorer** can load one real session on demand; results are cached so reruns do not re-download.
* **Strategy Lab** builds every legal strategy for your settings, simulates it and ranks it by
  `estimated time = Σ lap times + Σ pit-stop losses`. Lap times come from editable assumptions, a fit to the selected
  dataset, or the trained model. Nothing trains when the dashboard reruns.
* The first (default) lap-time assumptions are the **synthetic demonstration values** and are labelled as such.

## 11. Model evaluation methodology

Laps from one race are highly correlated, so a random lap-level split would leak information. PITWALL AI uses
`GroupShuffleSplit` by event: **no event appears in both training and test data**. Event identity is used only for
grouping, never as a feature. Metrics are MAE, RMSE (seconds) and R². The best eligible model must also beat the
baseline. Permutation importance is computed on the held-out events. Full details: [`docs/methodology.md`](docs/methodology.md).

**No performance numbers are quoted here on purpose**: any metrics shown by the app come from a training run you
performed on your own data.

## 12. Screenshots

> **Placeholder — no screenshots are included yet.** After running `streamlit run app.py`, capture each section
> and save the images to `docs/screenshots/` (e.g. `overview.png`, `strategy-lab.png`), then reference them here
> with `![Overview](docs/screenshots/overview.png)`. Only add screenshots of the real running app, and say whether
> they show demo or historical data.

## 13. Data sources and acknowledgements

* Historical timing data is accessed through the open-source **[FastF1](https://github.com/theOehrly/Fast-F1)**
  library (MIT licence), which retrieves data from public Formula 1 timing services. Please read FastF1's documentation
  for its data-source and usage notes.
* Formula 1 data and trademarks belong to their respective owners. **This repository does not include or redistribute
  any downloaded race data or cache** (they are git-ignored) — collect it yourself for personal, educational use.
* The app uses an original text-based identity and no official logos or imagery.

## 14. Known limitations

* Lap time depends on traffic, weather, track evolution, set-up and driver management; most are not modelled.
* The simulator treats the car as alone on track and ignores rivals, risk, Safety Cars (unless a hypothetical
  scenario is set) and tyre-set availability. The "best" strategy is best **under the stated assumptions** only.
* Pit-stop loss is a single user-set number; tyre wear is linear in age within a compound.
* Cleaning removes non-green and slow laps, which makes the data tidier than real racing.
* With few events, evaluation results are noisy and depend on the random split.
* Monte Carlo ranges describe simulated outcomes under invented uncertainty distributions, **not** confidence intervals.
* **Live FastF1 downloads were not exercised in the environment where this code was written** (no access to the data
  service); the loader's field mapping was checked against FastF1's documented columns, and all tests run offline.

## 15. Future improvements

SHAP explanations · gradient-boosted models (XGBoost) · richer tyre-wear curves with cliffs · traffic/undercut modelling ·
per-circuit pit-loss estimates · time-series (previous-lap) features · rolling-origin evaluation · FastAPI service ·
Docker image · SQLite storage for large collections.

## 16. Testing

```powershell
python -m pytest -q          # all tests (offline)
python -m ruff check .       # lint
```

The suite covers cleaning, demo data, the ML pipeline (including event-level separation and the "too few events"
case), the simulator, optimiser, dataset collection (with mocked FastF1) and a Streamlit smoke test of every page.
GitHub Actions (`.github/workflows/tests.yml`) runs lint and tests on pushes and pull requests.

## 17. Licence

MIT — see [`LICENSE`](LICENSE). Replace the placeholder copyright holder with your name before publishing.

## 18. Disclaimer

PITWALL AI is an independent educational fan project. It is not affiliated with, endorsed by, or connected to Formula
One, the FIA, any racing team or any driver, and it is not official or professional race-engineering software.
Predictions and simulations are estimates under stated assumptions.
