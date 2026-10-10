"""Lap-data normalisation and cleaning with explicit, documented rules.

Every rule is applied in a fixed order and the number of rows each rule removes is
reported, so nothing is dropped silently. The rules are described in
``docs/methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitwall_ai.config import DRY_COMPOUNDS, REQUIRED_COLUMNS


class DataValidationError(ValueError):
    """Raised when a dataset is missing required columns or is otherwise unusable."""


@dataclass(frozen=True)
class CleaningRules:
    """Thresholds used by :func:`clean_laps`."""

    min_lap_seconds: float = 40.0  # anything faster is a timing artefact
    max_lap_seconds: float = 200.0  # anything slower is not a representative lap
    max_ratio_to_event_median: float = 1.10  # drop laps > 110 % of the event median
    keep_compounds: tuple[str, ...] = DRY_COMPOUNDS
    exclude_first_lap: bool = True  # standing start + traffic: not representative
    green_flag_only: bool = True  # drop laps with yellow flags / SC / VSC / red flag


@dataclass
class CleaningReport:
    """How many rows each cleaning rule removed."""

    rows_in: int = 0
    rows_out: int = 0
    removed: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out, "removed": dict(self.removed)}


def to_seconds(series: pd.Series) -> pd.Series:
    """Convert lap durations to float seconds.

    Accepts ``timedelta64`` values, strings such as ``"0 days 00:01:32.456000"`` or
    ``"1:32.456"``, or numbers (assumed to be seconds already). Unparseable values
    become ``NaN`` rather than raising, so the cleaning step can count and drop them.
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def parse(value: object) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, pd.Timedelta):
            return value.total_seconds()
        text = str(value).strip()
        if not text:
            return np.nan
        if ":" in text and "days" not in text:
            # "m:ss.fff" or "h:mm:ss.fff"
            parts = text.split(":")
            try:
                seconds = 0.0
                for part in parts:
                    seconds = seconds * 60 + float(part)
                return seconds
            except ValueError:
                return np.nan
        try:
            return float(text)
        except ValueError:
            td = pd.to_timedelta(text, errors="coerce")
            return np.nan if pd.isna(td) else td.total_seconds()

    return series.map(parse).astype(float)


def validate_columns(df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    """Raise :class:`DataValidationError` if any required column is absent."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required column(s): {', '.join(missing)}")


def _truthy(series: pd.Series) -> pd.Series:
    """Coerce a boolean-like column (bool, 0/1, 'True'/'False', NaN) to bool."""
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "1.0", "yes"})


def make_event_id(season: int, round_number: int) -> str:
    """Stable identifier for one event, e.g. ``2023_05``."""
    return f"{int(season)}_{int(round_number):02d}"


def normalise_fastf1_laps(
    raw: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    event_name: str,
    session_type: str,
    data_source: str,
) -> pd.DataFrame:
    """Map a FastF1 ``session.laps`` table onto the project's canonical columns.

    Fields that FastF1 does not provide for a session are filled with neutral values
    (``NaN``/``False``) instead of crashing; :func:`clean_laps` then decides what to do
    with the affected rows.
    """
    required_raw = ("Driver", "LapNumber", "LapTime")
    missing = [c for c in required_raw if c not in raw.columns]
    if missing:
        raise DataValidationError(f"Raw lap data is missing column(s): {', '.join(missing)}")

    out = pd.DataFrame(index=raw.index)
    out["Season"] = int(season)
    out["RoundNumber"] = int(round_number)
    out["EventName"] = str(event_name)
    out["EventId"] = make_event_id(season, round_number)
    out["SessionType"] = str(session_type)
    out["Driver"] = raw["Driver"].astype("string")
    out["LapNumber"] = pd.to_numeric(raw["LapNumber"], errors="coerce")
    out["LapTimeSeconds"] = to_seconds(raw["LapTime"])
    out["Compound"] = (
        raw["Compound"].astype("string").str.upper() if "Compound" in raw.columns else pd.NA
    )
    out["TyreLife"] = (
        pd.to_numeric(raw["TyreLife"], errors="coerce") if "TyreLife" in raw.columns else np.nan
    )
    out["Stint"] = (
        pd.to_numeric(raw["Stint"], errors="coerce") if "Stint" in raw.columns else np.nan
    )

    pit_in = raw["PitInTime"].notna() if "PitInTime" in raw.columns else False
    pit_out = raw["PitOutTime"].notna() if "PitOutTime" in raw.columns else False
    out["IsPitLap"] = pit_in | pit_out

    out["TrackStatus"] = (
        raw["TrackStatus"].astype("string") if "TrackStatus" in raw.columns else pd.NA
    )
    out["IsAccurate"] = _truthy(raw["IsAccurate"]) if "IsAccurate" in raw.columns else True
    out["Deleted"] = _truthy(raw["Deleted"]) if "Deleted" in raw.columns else False
    out["DataSource"] = data_source
    return out.reset_index(drop=True)


def clean_laps(
    df: pd.DataFrame, rules: CleaningRules | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the documented cleaning rules and return ``(clean_df, report)``.

    Rules, in order:

    1. Missing, non-numeric or non-positive lap time.
    2. Lap time outside the plausible absolute range.
    3. Missing driver / lap number.
    4. Pit-in and pit-out laps (their times include the pit-lane).
    5. Deleted laps and laps flagged inaccurate by the timing system (if provided).
    6. Laps that are not fully green-flag (if ``TrackStatus`` is provided).
    7. First lap of the session (if enabled).
    8. Compounds outside the allowed (dry) set, or missing compound.
    9. Missing tyre age.
    10. Laps slower than ``max_ratio_to_event_median`` times the median of their event
        and session (traffic, mistakes, unreported slow laps).

    An empty input returns an empty frame and a report with zero counts.
    """
    rules = rules or CleaningRules()
    validate_columns(
        df,
        (
            "EventId",
            "SessionType",
            "Driver",
            "LapNumber",
            "LapTimeSeconds",
            "Compound",
            "TyreLife",
            "IsPitLap",
        ),
    )
    report = CleaningReport(rows_in=len(df))
    data = df.copy()

    def drop(rule: str, keep: pd.Series) -> None:
        nonlocal data
        keep = keep.fillna(False).astype(bool)
        report.removed[rule] = int((~keep).sum())
        data = data[keep]

    data["LapTimeSeconds"] = (
        to_seconds(data["LapTimeSeconds"]) if len(data) else data["LapTimeSeconds"]
    )
    drop("missing_or_non_positive_lap_time", data["LapTimeSeconds"] > 0)
    drop(
        "lap_time_outside_plausible_range",
        data["LapTimeSeconds"].between(rules.min_lap_seconds, rules.max_lap_seconds),
    )
    drop("missing_driver_or_lap_number", data["Driver"].notna() & data["LapNumber"].notna())
    drop("pit_in_or_out_lap", ~_truthy(data["IsPitLap"]))

    if "Deleted" in data.columns:
        drop("deleted_lap", ~_truthy(data["Deleted"]))
    if "IsAccurate" in data.columns:
        drop("inaccurate_timing", _truthy(data["IsAccurate"]))
    if rules.green_flag_only and "TrackStatus" in data.columns:
        status = data["TrackStatus"]
        # Only filter rows where a status is actually known.
        drop("not_green_flag", status.isna() | (status.astype("string").str.strip() == "1"))
    if rules.exclude_first_lap:
        drop("first_lap", data["LapNumber"] > 1)

    compound = data["Compound"].astype("string").str.upper()
    data["Compound"] = compound
    drop("compound_missing_or_not_allowed", compound.isin(rules.keep_compounds))
    drop("missing_tyre_age", pd.to_numeric(data["TyreLife"], errors="coerce").notna())

    if len(data):
        event_median = data.groupby(["EventId", "SessionType"])["LapTimeSeconds"].transform(
            "median"
        )
        drop(
            "slower_than_event_median_threshold",
            data["LapTimeSeconds"] <= event_median * rules.max_ratio_to_event_median,
        )
    else:
        report.removed["slower_than_event_median_threshold"] = 0

    data = data.reset_index(drop=True)
    data["LapNumber"] = data["LapNumber"].astype(int)
    data["TyreLife"] = data["TyreLife"].astype(float)
    report.rows_out = len(data)
    return data, report
