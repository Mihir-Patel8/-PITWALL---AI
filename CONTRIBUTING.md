# Contributing

Thanks for your interest in PITWALL AI! This is a student portfolio project, but suggestions and fixes are welcome.

## Set up

```
python -m venv .venv            # use `py -3.12 -m venv .venv` on Windows
# activate: .\.venv\Scripts\Activate.ps1 (Windows)  |  source .venv/bin/activate (macOS/Linux)
python -m pip install -r requirements-dev.txt
```

## Before you open a pull request

```
python -m ruff check .          # lint
python -m ruff format .         # formatting (line length 100)
python -m pytest -q             # tests (no internet required)
```

## Code structure

* `pitwall_ai/data/` - cleaning, demo data, FastF1 loading, dataset building
* `pitwall_ai/ml/` - features, training, grouped evaluation, prediction
* `pitwall_ai/strategy/` - estimators, simulator, optimiser, scenarios (no Streamlit imports)
* `pitwall_ai/views/` and `app.py` - Streamlit UI only
* `tests/` - offline tests using small synthetic fixtures

Rules of thumb: keep logic out of the UI, add a test with every behaviour change, never present synthetic data as
real, never commit downloaded datasets, caches or model files, and keep evaluation free of leakage (split by event).

## Proposing changes

Open an issue describing the problem or idea, then a pull request that references it. Keep pull requests focused.
