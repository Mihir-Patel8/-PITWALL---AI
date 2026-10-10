"""Load real historical lap data through the FastF1 library.

FastF1 downloads timing data from public Formula 1 data services on first use and keeps a
local cache so later calls are fast. Network access is therefore required the first time
an event is loaded. ``fastf1`` is imported lazily so the rest of the application (and the
test-suite) works without it being exercised.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

fastf1.set_log_level("DEBUG")

from pitwall_ai.config import (
    DATA_SOURCE_HISTORICAL,
    FASTF1_CACHE_DIR,
    SUPPORTED_SESSIONS,
)
from pitwall_ai.data.cleaning import (
    CleaningReport,
    CleaningRules,
    clean_laps,
    normalise_fastf1_laps,
)

logger = logging.getLogger(__name__)
_FIRST_SEASON_WITH_TYRE_DATA = 2018


class DataUnavailableError(RuntimeError):
    """Raised when historical data cannot be loaded (network, missing session, ...)."""


def enable_cache(cache_dir: Path = FASTF1_CACHE_DIR) -> Path:
    """Create the cache folder and point FastF1 at it."""
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def validate_request(season: int, session_type: str) -> None:
    """Check user-supplied season / session values before touching the network."""
    current_year = pd.Timestamp.now().year
    if not _FIRST_SEASON_WITH_TYRE_DATA <= int(season) <= current_year:
        raise ValueError(
            f"Season must be between {_FIRST_SEASON_WITH_TYRE_DATA} and {current_year} "
            "(tyre data is only reliably available from 2018)."
        )
    if session_type not in SUPPORTED_SESSIONS:
        raise ValueError(
            f"Unsupported session '{session_type}'. Choose from: {', '.join(SUPPORTED_SESSIONS)}"
        )


def get_event_schedule(season: int, cache_dir: Path = FASTF1_CACHE_DIR) -> pd.DataFrame:
    """Return the season schedule (RoundNumber, EventName, EventDate, Country, Location).

    Testing events are excluded. Raises :class:`DataUnavailableError` on failure.
    """
    validate_request(season, "R")
    import fastf1

    enable_cache(cache_dir)
    try:
        schedule = fastf1.get_event_schedule(int(season), include_testing=False)
    except Exception as exc:  # network / parsing errors from several libraries
        raise DataUnavailableError(
            f"Could not load the {season} event schedule: {exc}. "
            "Check your internet connection and try again."
        ) from exc
    columns = [
        c
        for c in ("RoundNumber", "EventName", "EventDate", "Country", "Location")
        if c in schedule.columns
    ]
    return schedule[columns].reset_index(drop=True)


def load_session_laps(
    season: int,
    round_number: int,
    session_type: str = "R",
    *,
    event_name: str | None = None,
    rules: CleaningRules | None = None,
    clean: bool = True,
    cache_dir: Path = FASTF1_CACHE_DIR,
) -> tuple[pd.DataFrame, CleaningReport | None]:
    """Load one session and return ``(laps, cleaning_report)``.

    With ``clean=False`` the normalised but uncleaned laps are returned (useful for
    showing pit laps in the Race Explorer) and the report is ``None``.

    Raises:
        ValueError: invalid season or session type.
        DataUnavailableError: the session could not be downloaded or has no lap data.
    """
    validate_request(season, session_type)
    import fastf1

    enable_cache(cache_dir)
    try:
        session = fastf1.get_session(int(season), int(round_number), session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        raw_laps = pd.DataFrame(session.laps)
        resolved_name = event_name or str(session.event["EventName"])
    except Exception as exc:
        raise DataUnavailableError(
            f"Could not load {season} round {round_number} ({session_type}): {exc}. "
            "This can be caused by no internet connection, a session that has not "
            "happened yet, or data that is not published for that session."
        ) from exc

    if raw_laps.empty:
        raise DataUnavailableError(
            f"{season} round {round_number} ({session_type}) was loaded but contains no laps."
        )

    normalised = normalise_fastf1_laps(
        raw_laps,
        season=season,
        round_number=round_number,
        event_name=resolved_name,
        session_type=session_type,
        data_source=DATA_SOURCE_HISTORICAL,
    )
    if not clean:
        return normalised, None
    cleaned, report = clean_laps(normalised, rules)
    logger.info("Loaded %s: %d -> %d laps", resolved_name, report.rows_in, report.rows_out)
    return cleaned, report
