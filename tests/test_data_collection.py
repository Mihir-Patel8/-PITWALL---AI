import pandas as pd
import pytest

from pitwall_ai.data import dataset_builder, fastf1_loader
from pitwall_ai.data.demo_data import generate_demo_laps


def fake_schedule(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RoundNumber": [1, 2, 3],
            "EventName": [f"{year} GP A", f"{year} GP B", f"{year} GP C"],
            "EventDate": pd.to_datetime([f"{year}-03-01", f"{year}-04-01", f"{year}-05-01"]),
        }
    )


def make_loader(fail_rounds=()):
    demo = generate_demo_laps(n_events=1, n_drivers=1)

    def loader(season, round_number, session_type, *, event_name=None, rules=None):
        if round_number in fail_rounds:
            raise fastf1_loader.DataUnavailableError("simulated network failure")
        return demo.assign(EventId=f"{season}_{round_number:02d}", Season=season), None

    return loader


TODAY = pd.Timestamp("2024-04-15")


def test_select_events_respects_max_events_and_skips_future():
    events = dataset_builder.select_events(
        2023, 2024, 4, schedule_loader=fake_schedule, today=TODAY
    )
    assert [e[:2] for e in events] == [(2023, 1), (2023, 2), (2023, 3), (2024, 1)]
    all_events = dataset_builder.select_events(
        2024, 2024, schedule_loader=fake_schedule, today=TODAY
    )
    assert [e[1] for e in all_events] == [1, 2]  # round 3 (May) is in the future


def test_select_events_validates_arguments():
    with pytest.raises(ValueError):
        dataset_builder.select_events(2024, 2023, schedule_loader=fake_schedule)
    with pytest.raises(ValueError):
        dataset_builder.select_events(2023, 2023, 0, schedule_loader=fake_schedule)


def test_failed_event_is_skipped_without_corrupting_output(tmp_path):
    out = tmp_path / "laps.csv"
    messages: list[str] = []
    result = dataset_builder.build_dataset(
        2023,
        2023,
        output_path=out,
        progress=messages.append,
        event_loader=make_loader(fail_rounds={2}),
        schedule_loader=fake_schedule,
    )
    assert len(result.events_ok) == 2 and len(result.events_failed) == 1
    saved = pd.read_csv(out)
    assert set(saved["EventId"]) == {"2023_01", "2023_03"} and len(saved) == result.rows
    assert any("FAILED" in m for m in messages)
    assert (tmp_path / "collection_report.json").exists() and not out.with_suffix(
        ".csv.tmp"
    ).exists()


def test_total_failure_does_not_overwrite_existing_dataset(tmp_path):
    out = tmp_path / "laps.csv"
    out.write_text("keep,me\n1,2\n")
    with pytest.raises(fastf1_loader.DataUnavailableError, match="No events could be loaded"):
        dataset_builder.build_dataset(
            2023,
            2023,
            output_path=out,
            progress=lambda m: None,
            event_loader=make_loader(fail_rounds={1, 2, 3}),
            schedule_loader=fake_schedule,
        )
    assert out.read_text() == "keep,me\n1,2\n"


@pytest.mark.parametrize(("season", "session"), [(2010, "R"), (2023, "XYZ")])
def test_loader_rejects_invalid_requests_before_network(season, session):
    with pytest.raises(ValueError):
        fastf1_loader.validate_request(season, session)


def test_collect_cli_reports_errors_cleanly(tmp_path, capsys, monkeypatch):
    from tests.conftest import load_script

    script = load_script("collect_data")

    def boom(*args, **kwargs):
        raise fastf1_loader.DataUnavailableError("offline")

    monkeypatch.setattr(script, "build_dataset", boom)
    code = script.main(
        ["--start-year", "2023", "--end-year", "2023", "--output", str(tmp_path / "x.csv")]
    )
    assert code == 1 and "offline" in capsys.readouterr().err
