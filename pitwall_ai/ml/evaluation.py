"""Event-grouped splitting, regression metrics and permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from pitwall_ai.config import RANDOM_SEED

MIN_EVENTS_FOR_EVALUATION = 4


class InsufficientEventsError(ValueError):
    """Raised when there are too few distinct events for an honest grouped evaluation."""


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """MAE, RMSE (both in seconds) and R-squared (``None`` when not meaningful)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape.")
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 and np.var(y_true) > 0 else None
    return {"mae": mae, "rmse": rmse, "r2": r2}


def grouped_train_test_split(
    groups: pd.Series,
    *,
    test_size: float = 0.25,
    seed: int = RANDOM_SEED,
    min_events: int = MIN_EVENTS_FOR_EVALUATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions so that no event appears in both train and test.

    Raises:
        InsufficientEventsError: fewer than ``min_events`` distinct events are available.
            No weaker fallback (e.g. a random lap split) is attempted because it would
            leak same-race information and overstate generalisation.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    n_events = groups.nunique()
    if n_events < min_events:
        raise InsufficientEventsError(
            f"Only {n_events} distinct event(s) found; at least {min_events} are needed for a "
            "meaningful event-level train/test split. Collect more events, e.g. "
            "`python scripts/collect_data.py --start-year 2023 --end-year 2024 --max-events 8`."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups.to_numpy()))
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:  # defensive: should be impossible
        raise RuntimeError(f"Event leakage detected between train and test: {sorted(overlap)}")
    return train_idx, test_idx


def permutation_importance_table(
    pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_repeats: int = 5,
    seed: int = RANDOM_SEED,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """Permutation importance of each raw input on **held-out** data (increase in MAE)."""
    if len(x_test) > max_rows:
        sample = x_test.sample(max_rows, random_state=seed).index
        x_test, y_test = x_test.loc[sample], y_test.loc[sample]
    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(x_test.columns),
            "mae_increase_seconds": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return table.sort_values("mae_increase_seconds", ascending=False).reset_index(drop=True)
