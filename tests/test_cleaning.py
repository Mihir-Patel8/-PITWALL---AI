import numpy as np
import pandas as pd
import pytest

from pitwall_ai.data.cleaning import (
    CleaningRules,
    DataValidationError,
    clean_laps,
    normalise_fastf1_laps,
    to_seconds,
    validate_columns,
)


def base_rows(n: int = 12) -> list[dict]:
    return [
        {
            "EventId": "e1",
            "SessionType": "R",
            "Driver": "AAA",
            "LapNumber": i + 2,
            "LapTimeSeconds": 90.0 + 0.1 * i,
            "Compound": "MEDIUM",
            "TyreLife": float(i + 1),
            "IsPitLap": False,
            "TrackStatus": "1",
            "IsAccurate": True,
            "Deleted": False,
        }
        for i in range(n)
    ]


def with_bad(**overrides) -> dict:
    row = base_rows(1)[0] | overrides
    return row


def test_to_seconds_timedelta():
    series = pd.Series(pd.to_timedelta(["0 days 00:01:32.456", None]))
    result = to_seconds(series)
    assert result.iloc[0] == pytest.approx(92.456)
    assert np.isnan(result.iloc[1])


def test_to_seconds_strings_and_numbers():
    result = to_seconds(pd.Series(["1:32.456", "0 days 00:01:30.000000", "", "abc", None]))
    assert result.iloc[0] == pytest.approx(92.456)
    assert result.iloc[1] == pytest.approx(90.0)
    assert result.iloc[2:].isna().all()
    assert to_seconds(pd.Series([90, 91.5])).tolist() == [90.0, 91.5]


def test_each_rule_removes_exactly_its_row():
    rows = base_rows()
    rows += [
        with_bad(LapTimeSeconds=np.nan),
        with_bad(LapTimeSeconds=0.0),
        with_bad(LapTimeSeconds=20.0),  # implausibly fast
        with_bad(IsPitLap=True),
        with_bad(Deleted=True),
        with_bad(IsAccurate=False),
        with_bad(TrackStatus="4"),
        with_bad(LapNumber=1),
        with_bad(Compound="WET"),
        with_bad(TyreLife=np.nan),
        with_bad(LapTimeSeconds=120.0),  # > 110 % of event median
    ]
    clean, report = clean_laps(pd.DataFrame(rows))
    assert report.rows_in == 23 and report.rows_out == 12 == len(clean)
    expected = {
        "missing_or_non_positive_lap_time": 2,
        "lap_time_outside_plausible_range": 1,
        "pit_in_or_out_lap": 1,
        "deleted_lap": 1,
        "inaccurate_timing": 1,
        "not_green_flag": 1,
        "first_lap": 1,
        "compound_missing_or_not_allowed": 1,
        "missing_tyre_age": 1,
        "slower_than_event_median_threshold": 1,
    }
    for rule, count in expected.items():
        assert report.removed[rule] == count, rule
    assert clean["LapTimeSeconds"].between(90, 92).all()


def test_optional_columns_may_be_absent():
    frame = pd.DataFrame(base_rows()).drop(columns=["TrackStatus", "IsAccurate", "Deleted"])
    clean, report = clean_laps(frame)
    assert len(clean) == 12 and "not_green_flag" not in report.removed


def test_rules_are_configurable():
    frame = pd.DataFrame(base_rows() + [with_bad(LapNumber=1)])
    clean, _ = clean_laps(frame, CleaningRules(exclude_first_lap=False))
    assert len(clean) == 13


def test_required_columns_validated():
    with pytest.raises(DataValidationError, match="Driver"):
        clean_laps(pd.DataFrame(base_rows()).drop(columns=["Driver"]))
    with pytest.raises(DataValidationError):
        validate_columns(pd.DataFrame({"a": [1]}))


def test_empty_dataset_is_handled():
    empty = pd.DataFrame(columns=list(base_rows(1)[0]))
    clean, report = clean_laps(empty)
    assert clean.empty and report.rows_in == 0 and report.rows_out == 0


def test_normalise_fastf1_laps_maps_fields():
    raw = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "HAM"],
            "LapNumber": [1.0, 2.0, 2.0],
            "LapTime": pd.to_timedelta(["0 days 00:01:40", "0 days 00:01:30.500", None]),
            "Compound": ["soft", "SOFT", None],
            "TyreLife": [1.0, 2.0, np.nan],
            "PitOutTime": pd.to_timedelta(["0 days 00:00:05", None, None]),
        }
    )
    out = normalise_fastf1_laps(
        raw,
        season=2023,
        round_number=5,
        event_name="Test GP",
        session_type="R",
        data_source="fastf1",
    )
    assert out["EventId"].unique().tolist() == ["2023_05"]
    assert out["LapTimeSeconds"].iloc[1] == pytest.approx(90.5)
    assert out["IsPitLap"].tolist() == [True, False, False]
    assert out["Compound"].iloc[0] == "SOFT"
    assert out["DataSource"].unique().tolist() == ["fastf1"]


def test_normalise_requires_core_columns():
    with pytest.raises(DataValidationError):
        normalise_fastf1_laps(
            pd.DataFrame({"Driver": ["A"]}),
            season=2023,
            round_number=1,
            event_name="x",
            session_type="R",
            data_source="fastf1",
        )
