"""Train, compare and persist lap-time models using event-level evaluation."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pitwall_ai.config import (
    ARTIFACTS_DIR,
    DATA_SOURCE_SYNTHETIC,
    RANDOM_SEED,
)
from pitwall_ai.ml import evaluation, features

BASELINE_NAME = "baseline_mean"
ELIGIBLE_MODELS = ("linear_regression", "random_forest")


class NoEligibleModelError(RuntimeError):
    """Raised when no real model beats the baseline on held-out events."""


def build_models(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Return the candidate models, each as ``preprocessing -> estimator``."""
    return {
        BASELINE_NAME: Pipeline(
            [("prep", features.build_preprocessor()), ("model", DummyRegressor(strategy="mean"))]
        ),
        "linear_regression": Pipeline(
            [("prep", features.build_preprocessor()), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [
                ("prep", features.build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=seed
                    ),
                ),
            ]
        ),
    }


@dataclass
class TrainingResult:
    """Everything produced by :func:`train_and_evaluate`."""

    fitted_models: dict[str, Pipeline]
    metrics: dict[str, dict[str, float | None]]
    best_model_name: str | None
    report: dict[str, object]
    test_predictions: pd.DataFrame
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    seed: int = RANDOM_SEED,
    min_events: int = evaluation.MIN_EVENTS_FOR_EVALUATION,
    n_importance_repeats: int = 5,
) -> TrainingResult:
    """Fit all candidates on training events and score them on unseen events.

    The best *eligible* model is the one with the lowest held-out MAE among the real
    models, provided it also beats the baseline MAE. Otherwise ``best_model_name`` is
    ``None`` and the report says so.

    Raises:
        DataValidationError: invalid or missing data.
        InsufficientEventsError: too few distinct events for grouped evaluation.
    """
    x, y, groups = features.split_xy(df)
    train_idx, test_idx = evaluation.grouped_train_test_split(
        groups, test_size=test_size, seed=seed, min_events=min_events
    )
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float | None]] = {}
    predictions = pd.DataFrame(
        {
            "EventId": groups.iloc[test_idx].to_numpy(),
            "Driver": x_test["Driver"].to_numpy(),
            "Compound": x_test["Compound"].to_numpy(),
            "TyreLife": x_test["TyreLife"].to_numpy(),
            "LapNumber": x_test["LapNumber"].to_numpy(),
            "observed": y_test.to_numpy(),
        }
    )
    for name, pipeline in build_models(seed).items():
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        if not np.all(np.isfinite(y_pred)):
            raise RuntimeError(f"Model '{name}' produced non-finite predictions.")
        fitted[name] = pipeline
        metrics[name] = evaluation.regression_metrics(y_test.to_numpy(), y_pred)
        predictions[f"pred_{name}"] = y_pred

    baseline_mae = metrics[BASELINE_NAME]["mae"]
    candidates = {
        name: metrics[name]["mae"]
        for name in ELIGIBLE_MODELS
        if metrics[name]["mae"] < baseline_mae
    }
    best = min(candidates, key=candidates.get) if candidates else None

    importance = pd.DataFrame()
    if best is not None:
        importance = evaluation.permutation_importance_table(
            fitted[best], x_test, y_test, n_repeats=n_importance_repeats, seed=seed
        )

    data_source = (
        sorted(df["DataSource"].astype(str).unique().tolist())
        if "DataSource" in df.columns
        else ["unknown"]
    )
    report: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_source": data_source,
        "is_synthetic": data_source == [DATA_SOURCE_SYNTHETIC],
        "split_method": "GroupShuffleSplit by EventId (no event in both train and test)",
        "test_size_fraction_of_events": test_size,
        "random_seed": seed,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_idx)),
        "n_rows_test": int(len(test_idx)),
        "n_events_total": int(groups.nunique()),
        "n_events_train": int(groups.iloc[train_idx].nunique()),
        "n_events_test": int(groups.iloc[test_idx].nunique()),
        "test_event_ids": sorted(groups.iloc[test_idx].unique().tolist()),
        "features": list(features.FEATURE_COLUMNS),
        "target": features.TARGET,
        "metrics": metrics,
        "best_model": best,
        "selection_rule": "lowest held-out MAE among eligible models that also beat the baseline",
        "model_trained_on": "training events only (not refit on the held-out events)",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "importance": importance.to_dict(orient="records"),
        "importance_note": (
            "Permutation importance on held-out events: how much MAE (seconds) increases when "
            "one input is shuffled. Correlated inputs share importance, so treat as indicative."
        ),
    }
    return TrainingResult(fitted, metrics, best, report, predictions, importance)


def save_artifacts(result: TrainingResult, output_dir: Path = ARTIFACTS_DIR) -> dict[str, Path]:
    """Write the evaluation report, held-out predictions and (if eligible) the model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output_dir / "evaluation.json",
        "predictions": output_dir / "test_predictions.csv",
    }
    paths["evaluation"].write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    sample = result.test_predictions
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=RANDOM_SEED)
    sample.to_csv(paths["predictions"], index=False)

    model_path = output_dir / "lap_time_model.joblib"
    if result.best_model_name is not None:
        joblib.dump(
            {
                "pipeline": result.fitted_models[result.best_model_name],
                "model_name": result.best_model_name,
                "features": list(features.FEATURE_COLUMNS),
                "report": result.report,
            },
            model_path,
        )
        paths["model"] = model_path
    elif model_path.exists():
        model_path.unlink()  # never leave a stale model next to a report that rejects it
    return paths
