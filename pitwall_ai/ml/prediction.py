"""Load a saved model and make validated lap-time predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pitwall_ai.config import EVALUATION_PATH, MODEL_PATH
from pitwall_ai.ml import features


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained model file exists."""


@dataclass
class LapTimePredictor:
    """Wrapper around a fitted scikit-learn pipeline."""

    pipeline: object
    model_name: str
    report: dict[str, object]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict lap time in seconds for rows with the model's input columns.

        Raises:
            DataValidationError: missing columns or invalid values.
            ValueError: if the model returns non-finite output.
        """
        features.validate_feature_frame(frame, require_target=False)
        values = self.pipeline.predict(frame[list(features.FEATURE_COLUMNS)])
        if not np.all(np.isfinite(values)):
            raise ValueError("Model returned non-finite predictions.")
        return np.asarray(values, dtype=float)


def load_predictor(model_path: Path = MODEL_PATH) -> LapTimePredictor:
    """Load the model saved by ``scripts/train_model.py``."""
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model at {model_path}. Run: "
            "python scripts/train_model.py --input data/processed/laps.csv"
        )
    bundle = joblib.load(model_path)
    return LapTimePredictor(bundle["pipeline"], bundle["model_name"], bundle["report"])


def load_evaluation_report(path: Path = EVALUATION_PATH) -> dict[str, object] | None:
    """Return the saved evaluation JSON, or ``None`` if it does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
