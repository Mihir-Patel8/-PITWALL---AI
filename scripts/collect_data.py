"""Collect historical F1 laps via FastF1 and save a processed CSV.

Example (run from the repository root):

    python scripts/collect_data.py --start-year 2023 --end-year 2024 --max-events 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall_ai.config import (  # noqa: E402
    DEFAULT_COLLECTION_SESSIONS,
    DEFAULT_DATASET_PATH,
    SUPPORTED_SESSIONS,
)
from pitwall_ai.data.dataset_builder import build_dataset  # noqa: E402
from pitwall_ai.data.fastf1_loader import DataUnavailableError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect historical F1 lap data with FastF1.")
    parser.add_argument("--start-year", type=int, required=True, help="First season (>= 2018).")
    parser.add_argument("--end-year", type=int, required=True, help="Last season (inclusive).")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after this many events in total (useful for quick tests).",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=list(DEFAULT_COLLECTION_SESSIONS),
        choices=list(SUPPORTED_SESSIONS),
        help="Session types (default: R).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Output CSV (default: {DEFAULT_DATASET_PATH.name} in data/processed).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "Collecting data. The first download needs an internet connection and can take a while;"
        " later runs reuse the local FastF1 cache."
    )
    try:
        result = build_dataset(
            args.start_year,
            args.end_year,
            output_path=args.output,
            max_events=args.max_events,
            sessions=tuple(args.sessions),
        )
    except (DataUnavailableError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Done: {result.rows} laps, {len(result.events_ok)} sessions ok, "
        f"{len(result.events_failed)} failed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
