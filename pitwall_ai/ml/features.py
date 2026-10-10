"""Feature definitions and preprocessing for lap-time prediction.

Only information that is known *before* the lap starts is used:

* ``Compound``  - the tyre on the car,
* ``TyreLife``  - laps already completed on that set (tyre age),
* ``LapNumber`` - how far into the race we are (proxy for fuel load),
* ``Driver``    - driver identifier (captures a persistent pace offset),
* ``EventName`` - circuit/event identifier (captures track length / base pace).

The *event id* (``EventId``) is used only to group laps for train/test splitting; it is
never a model input. Sector times and the target lap time are never used.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pitwall_ai.data.cleaning import DataValidationError

TARGET = "LapTimeSeconds"
GROUP_COLUMN = "EventId"
NUMERIC_FEATURES: tuple[str, ...] = ("TyreLife", "LapNumber")
CATEGORICAL_FEATURES: tuple[str, ...] = ("Compound", "Driver", "EventName")
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TRAINING_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET, GROUP_COLUMN)


def validate_feature_frame(df: pd.DataFrame, *, require_target: bool) -> None:
    """Check columns and basic value validity; raise :class:`DataValidationError`."""
    needed = TRAINING_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise DataValidationError("The dataset is empty.")

    if (pd.to_numeric(df["LapNumber"], errors="coerce") < 1).any():
        raise DataValidationError("LapNumber must be >= 1 for every row.")
    tyre_life = pd.to_numeric(df["TyreLife"], errors="coerce")
    if (tyre_life < 0).any():
        raise DataValidationError("TyreLife must not be negative.")
    if require_target:
        target = pd.to_numeric(df[TARGET], errors="coerce")
        if target.isna().any() or (target <= 0).any():
            raise DataValidationError(f"{TARGET} must be a positive number for every row.")


def build_preprocessor() -> ColumnTransformer:
    """Imputation + one-hot encoding; unseen categories are ignored at prediction time."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, groups)`` after validation."""
    validate_feature_frame(df, require_target=True)
    features = df[list(FEATURE_COLUMNS)].copy()
    features["Compound"] = features["Compound"].astype("object")
    features["Driver"] = features["Driver"].astype("object")
    features["EventName"] = features["EventName"].astype("object")
    return features, df[TARGET].astype(float), df[GROUP_COLUMN].astype(str)
