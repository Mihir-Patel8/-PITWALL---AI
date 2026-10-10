"""Central configuration: paths, constants, schema and colours.

Everything that several modules need to agree on lives here.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
FASTF1_CACHE_DIR: Path = RAW_DIR / "fastf1_cache"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

DEFAULT_DATASET_PATH: Path = PROJECT_ROOT / "laps.csv"
COLLECTION_REPORT_PATH: Path = PROCESSED_DIR / "collection_report.json"
MODEL_PATH: Path = ARTIFACTS_DIR / "lap_time_model.joblib"
EVALUATION_PATH: Path = ARTIFACTS_DIR / "evaluation.json"
TEST_PREDICTIONS_PATH: Path = ARTIFACTS_DIR / "test_predictions.csv"

# --------------------------------------------------------------------------- reproducibility
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- data schema
DRY_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS: tuple[str, ...] = ("INTERMEDIATE", "WET")

DATA_SOURCE_HISTORICAL = "fastf1"
DATA_SOURCE_SYNTHETIC = "synthetic"

#: Columns every cleaned lap table must contain.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Season",
    "RoundNumber",
    "EventName",
    "EventId",
    "SessionType",
    "Driver",
    "LapNumber",
    "LapTimeSeconds",
    "Compound",
    "TyreLife",
    "Stint",
    "IsPitLap",
    "DataSource",
)

# --------------------------------------------------------------------------- session types
#: Session types supported by the Race Explorer (FastF1 identifiers).
SUPPORTED_SESSIONS: dict[str, str] = {
    "R": "Race",
    "S": "Sprint",
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
}
DEFAULT_COLLECTION_SESSIONS: tuple[str, ...] = ("R",)

# --------------------------------------------------------------------------- strategy defaults
DEFAULT_PIT_LOSS_SECONDS: float = 21.0  # hypothetical default; varies a lot by circuit
DEFAULT_MONTE_CARLO_RUNS: int = 500

# --------------------------------------------------------------------------- UI
APP_NAME = "PITWALL AI"
APP_TAGLINE = "Formula 1 Race Strategy Intelligence System"

COLORS: dict[str, str] = {
    "background": "#0B0F1A",
    "panel": "#141B2D",
    "grid": "#26304A",
    "text": "#F4F6FA",
    "muted": "#9AA6C2",
    "red": "#E10600",
    "cyan": "#00D2FF",
    "teal": "#14B8A6",
}

#: Tyre colours are generic (red/yellow/white-ish), not copied from any official branding.
COMPOUND_COLORS: dict[str, str] = {
    "SOFT": "#FF4D4D",
    "MEDIUM": "#FFD23F",
    "HARD": "#E8ECF4",
    "INTERMEDIATE": "#3DDC84",
    "WET": "#3B82F6",
    "UNKNOWN": "#7A869E",
}

DISCLAIMER = (
    "PITWALL AI is an independent, educational fan project. It is not affiliated with, "
    "endorsed by, or connected to Formula One, the FIA, any racing team or any driver. "
    "Predictions and simulations are estimates under stated assumptions and are not "
    "professional race-engineering advice."
)
