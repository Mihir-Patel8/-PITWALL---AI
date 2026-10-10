import pandas as pd
import pytest

from pitwall_ai.config import REQUIRED_COLUMNS
from pitwall_ai.data.demo_data import DEMO_LABEL, generate_demo_laps, is_synthetic


def test_fixed_seed_is_reproducible():
    pd.testing.assert_frame_equal(generate_demo_laps(seed=1), generate_demo_laps(seed=1))


def test_different_seed_changes_data():
    a, b = generate_demo_laps(seed=1), generate_demo_laps(seed=2)
    assert not a["LapTimeSeconds"].equals(b["LapTimeSeconds"])


def test_expected_columns_and_valid_values():
    frame = generate_demo_laps(n_events=3, n_drivers=4)
    assert set(REQUIRED_COLUMNS) <= set(frame.columns)
    assert (frame["LapTimeSeconds"] > 0).all()
    assert (frame["TyreLife"] >= 1).all() and (frame["LapNumber"] >= 1).all()
    assert frame["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).all()
    assert frame["Driver"].nunique() == 4 and frame["EventId"].nunique() == 3


def test_tyre_age_resets_after_each_pit_stop():
    frame = generate_demo_laps(n_events=1, n_drivers=1)
    stint_starts = frame.groupby("Stint")["TyreLife"].min()
    assert (stint_starts == 1).all()


def test_dataset_is_clearly_synthetic():
    frame = generate_demo_laps(n_events=2, n_drivers=2)
    assert (frame["DataSource"] == "synthetic").all()
    assert is_synthetic(frame)
    assert "synthetic" in DEMO_LABEL.lower() and frame.attrs["label"] == DEMO_LABEL
    assert not is_synthetic(frame.assign(DataSource="fastf1"))
    assert not is_synthetic(frame.iloc[0:0])


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        generate_demo_laps(n_events=0)
    with pytest.raises(ValueError):
        generate_demo_laps(n_drivers=999)
