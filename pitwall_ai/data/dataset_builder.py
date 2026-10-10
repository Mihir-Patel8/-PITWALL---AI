"""Collect many historical events into one processed CSV."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pitwall_ai.config import (
    DEFAULT_COLLECTION_SESSIONS,
    DEFAULT_DATASET_PATH,
)
from pitwall_ai.data import fastf1_loader
from pitwall_ai.data.cleaning import CleaningRules

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str], None]


@dataclass
class CollectionResult:
    """Outcome of a collection run."""

    output_path: Path
    rows: int = 0
    events_ok: list[str] = field(default_factory=list)
    events_failed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "rows": self.rows,
            "events_ok": self.events_ok,
            "events_failed": self.events_failed,
        }


def select_events(
    start_year: int,
    end_year: int,
    max_events: int | None = None,
    *,
    schedule_loader: Callable[[int], pd.DataFrame] = fastf1_loader.get_event_schedule,
    today: pd.Timestamp | None = None,
) -> list[tuple[int, int, str]]:
    """Choose ``(season, round, event_name)`` tuples from the official FastF1 schedules.

    Only events whose date has already passed are returned. ``max_events`` caps the total
    number of events (taken in chronological order) which is handy for quick tests.
    """
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be at least 1")
    today = today or pd.Timestamp.now().normalize()

    selected: list[tuple[int, int, str]] = []
    for year in range(start_year, end_year + 1):
        schedule = schedule_loader(year)
        for _, row in schedule.iterrows():
            event_date = pd.to_datetime(row.get("EventDate"), errors="coerce")
            if pd.notna(event_date) and event_date.tz_localize(None) > today:
                continue  # future event: nothing to download
            selected.append((year, int(row["RoundNumber"]), str(row["EventName"])))
            if max_events is not None and len(selected) >= max_events:
                return selected
    return selected


def build_dataset(
    start_year: int,
    end_year: int,
    *,
    output_path: Path = DEFAULT_DATASET_PATH,
    max_events: int | None = None,
    sessions: tuple[str, ...] = DEFAULT_COLLECTION_SESSIONS,
    rules: CleaningRules | None = None,
    progress: ProgressFn = print,
    event_loader: Callable[..., tuple[pd.DataFrame, object]] = fastf1_loader.load_session_laps,
    schedule_loader: Callable[[int], pd.DataFrame] = fastf1_loader.get_event_schedule,
) -> CollectionResult:
    """Download, clean and save laps for a range of seasons.

    A failing event is logged and skipped; successful events are never affected. The CSV is
    written to a temporary file and moved into place only if at least one event succeeded,
    so a failed run cannot corrupt an existing dataset.
    """
    events = select_events(start_year, end_year, max_events, schedule_loader=schedule_loader)
    if not events:
        raise fastf1_loader.DataUnavailableError("No completed events found for that range.")

    result = CollectionResult(output_path=output_path)
    frames: list[pd.DataFrame] = []
    total = len(events) * len(sessions)
    step = 0
    for season, round_number, name in events:
        for session_type in sessions:
            step += 1
            label = f"{season} R{round_number:02d} {name} [{session_type}]"
            progress(f"[{step}/{total}] Loading {label} ...")
            try:
                laps, report = event_loader(
                    season, round_number, session_type, event_name=name, rules=rules
                )
            except (fastf1_loader.DataUnavailableError, ValueError) as exc:
                result.events_failed[label] = str(exc)
                progress(f"    FAILED: {exc}")
                continue
            if laps.empty:
                result.events_failed[label] = "no eligible laps after cleaning"
                progress("    skipped: no eligible laps after cleaning")
                continue
            frames.append(laps)
            result.events_ok.append(label)
            progress(f"    ok: {len(laps)} eligible laps")

    if not frames:
        raise fastf1_loader.DataUnavailableError(
            "No events could be loaded, so no dataset was written. "
            f"Failures: {result.events_failed}"
        )

    combined = pd.concat(frames, ignore_index=True)
    result.rows = len(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    combined.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)

    report_path = output_path.with_name("collection_report.json")
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    progress(f"Saved {result.rows} laps from {len(result.events_ok)} sessions to {output_path}")
    return result
