"""Train and evaluate lap-time models on a collected dataset.

Example (run from the repository root):

    python scripts/train_model.py --input data/processed/laps.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from pitwall_ai.config import ARTIFACTS_DIR, DATA_SOURCE_SYNTHETIC, DEFAULT_DATASET_PATH  # noqa: E402
from pitwall_ai.data.cleaning import DataValidationError  # noqa: E402
from pitwall_ai.ml import features, training  # noqa: E402
from pitwall_ai.ml.evaluation import InsufficientEventsError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PITWALL AI lap-time models.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Processed laps CSV from collect_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Where to save the model and evaluation report.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of EVENTS held out for testing (default 0.25).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit training on synthetic demo data (for software testing only).",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path, allow_synthetic: bool) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(
            f"Input file not found: {path}. Collect data first, e.g. "
            "`python scripts/collect_data.py --start-year 2023 --end-year 2024 --max-events 4`."
        )
    df = pd.read_csv(path)
    missing = [c for c in features.TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Input file is missing column(s): {', '.join(missing)}")
    before = len(df)
    df = df.dropna(subset=[features.TARGET, "TyreLife", "LapNumber", "EventId"])
    if len(df) < before:
        print(f"Note: dropped {before - len(df)} rows with missing target/tyre age/lap/event.")
    if (
        "DataSource" in df.columns
        and (df["DataSource"] == DATA_SOURCE_SYNTHETIC).any()
        and not allow_synthetic
    ):
        raise DataValidationError(
            "The input contains SYNTHETIC demo data. Training on it says nothing about real F1 "
            "races. Use real data, or pass --allow-synthetic to test the software."
        )
    return df


def print_summary(report: dict[str, object]) -> None:
    print("\n=== PITWALL AI training summary ===")
    print(
        f"Data source(s): {report['data_source']}"
        + ("   [SYNTHETIC - not real F1 data]" if report["is_synthetic"] else "")
    )
    print(
        f"Events: {report['n_events_train']} train / {report['n_events_test']} test "
        f"| Laps: {report['n_rows_train']} train / {report['n_rows_test']} test"
    )
    print(f"Split: {report['split_method']}")
    print(f"{'model':<20}{'MAE (s)':>10}{'RMSE (s)':>10}{'R2':>8}")
    for name, m in report["metrics"].items():
        r2 = "n/a" if m["r2"] is None else f"{m['r2']:.3f}"
        print(f"{name:<20}{m['mae']:>10.3f}{m['rmse']:>10.3f}{r2:>8}")
    print(f"Best eligible model: {report['best_model'] or 'NONE (no model beat the baseline)'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = load_dataset(args.input, args.allow_synthetic)
        result = training.train_and_evaluate(df, test_size=args.test_size, seed=args.seed)
    except (DataValidationError, InsufficientEventsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    paths = training.save_artifacts(result, args.output_dir)
    print_summary(result.report)
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    if result.best_model_name is None:
        print(
            "No model was saved because none beat the baseline on held-out events.", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
