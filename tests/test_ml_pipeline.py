import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pitwall_ai.data.cleaning import DataValidationError
from pitwall_ai.ml import evaluation, features, training
from pitwall_ai.ml.evaluation import InsufficientEventsError
from pitwall_ai.ml.prediction import LapTimePredictor, ModelNotFoundError, load_predictor
from tests.conftest import load_script


@pytest.fixture(scope="module")
def trained(demo_clean):
    return training.train_and_evaluate(demo_clean, n_importance_repeats=2)


def test_preprocessing_outputs_finite_matrix_and_handles_unseen_categories(demo_clean):
    x, _, _ = features.split_xy(demo_clean)
    prep = features.build_preprocessor().fit(x)
    unseen = x.head(3).copy()
    unseen["Driver"] = "NEW_DRIVER"
    unseen["EventName"] = "Brand New Circuit"
    unseen.loc[unseen.index[0], "TyreLife"] = np.nan  # missing numeric value
    out = prep.transform(unseen)
    assert out.shape[0] == 3 and np.isfinite(out).all()


def test_features_exclude_target_and_event_id(demo_clean):
    x, y, groups = features.split_xy(demo_clean)
    assert features.TARGET not in x.columns and features.GROUP_COLUMN not in x.columns
    assert set(x.columns) == set(features.FEATURE_COLUMNS)
    assert len(y) == len(groups) == len(x)


def test_grouped_split_keeps_events_apart(demo_clean):
    groups = demo_clean["EventId"].astype(str)
    train, test = evaluation.grouped_train_test_split(groups, seed=3)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    assert len(train) + len(test) == len(groups)


def test_too_few_events_is_reported_honestly(demo_clean):
    two = demo_clean[demo_clean["EventId"].isin(demo_clean["EventId"].unique()[:2])]
    with pytest.raises(InsufficientEventsError, match="at least 4"):
        training.train_and_evaluate(two)


def test_regression_metrics_known_values():
    m = evaluation.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert m["r2"] == pytest.approx(-1.0)
    assert evaluation.regression_metrics(np.array([5.0, 5.0]), np.array([5.0, 6.0]))["r2"] is None
    with pytest.raises(ValueError):
        evaluation.regression_metrics(np.array([]), np.array([]))


def test_training_report_is_consistent(trained):
    report = trained.report
    assert set(trained.metrics) == {"baseline_mean", "linear_regression", "random_forest"}
    assert all(np.isfinite(m["mae"]) and np.isfinite(m["rmse"]) for m in trained.metrics.values())
    assert report["is_synthetic"] is True
    assert report["n_events_train"] + report["n_events_test"] == report["n_events_total"]
    assert report["n_rows_train"] + report["n_rows_test"] == report["n_rows_total"]
    assert trained.best_model_name in training.ELIGIBLE_MODELS
    best_mae = trained.metrics[trained.best_model_name]["mae"]
    assert best_mae < trained.metrics["baseline_mean"]["mae"]
    assert not trained.importance.empty


def test_save_and_predict_roundtrip(trained, tmp_path):
    paths = training.save_artifacts(trained, tmp_path)
    assert {"evaluation", "predictions", "model"} <= set(paths)
    assert json.loads(paths["evaluation"].read_text())["best_model"] == trained.best_model_name
    predictor = load_predictor(paths["model"])
    frame = pd.DataFrame(
        {
            "TyreLife": [1, 10],
            "LapNumber": [5, 30],
            "Compound": ["SOFT", "HARD"],
            "Driver": ["DR01", "DR02"],
            "EventName": ["Demo Circuit Alpha"] * 2,
        }
    )
    pred = predictor.predict(frame)
    assert pred.shape == (2,) and np.isfinite(pred).all() and (pred > 0).all()


def test_stale_model_removed_when_nothing_beats_baseline(trained, tmp_path):
    (tmp_path / "lap_time_model.joblib").write_bytes(b"stale")
    rejected = training.TrainingResult(
        trained.fitted_models, trained.metrics, None, dict(trained.report), trained.test_predictions
    )
    paths = training.save_artifacts(rejected, tmp_path)
    assert "model" not in paths and not (tmp_path / "lap_time_model.joblib").exists()


def test_invalid_training_data_gives_useful_errors(demo_clean):
    with pytest.raises(DataValidationError, match="Missing required column"):
        features.split_xy(demo_clean.drop(columns=["Compound"]))
    with pytest.raises(DataValidationError, match="LapTimeSeconds"):
        features.split_xy(demo_clean.assign(LapTimeSeconds=-1.0))
    with pytest.raises(DataValidationError, match="TyreLife"):
        features.split_xy(demo_clean.assign(TyreLife=-2.0))
    with pytest.raises(DataValidationError, match="empty"):
        features.split_xy(demo_clean.iloc[0:0])


def test_predictor_validates_input_and_missing_model(trained, tmp_path):
    predictor = LapTimePredictor(
        trained.fitted_models["linear_regression"], "linear_regression", {}
    )
    with pytest.raises(DataValidationError):
        predictor.predict(pd.DataFrame({"TyreLife": [1]}))
    with pytest.raises(ModelNotFoundError, match="train_model.py"):
        load_predictor(tmp_path / "missing.joblib")


def test_saved_bundle_contains_metadata(trained, tmp_path):
    path = training.save_artifacts(trained, tmp_path)["model"]
    bundle = joblib.load(path)
    assert bundle["report"]["split_method"].startswith("GroupShuffleSplit")
    assert bundle["features"] == list(features.FEATURE_COLUMNS)


def test_train_cli_refuses_synthetic_data_by_default(demo_clean, tmp_path, capsys):
    csv = tmp_path / "laps.csv"
    demo_clean.to_csv(csv, index=False)
    script = load_script("train_model")
    assert script.main(["--input", str(csv), "--output-dir", str(tmp_path / "out")]) == 1
    assert "SYNTHETIC" in capsys.readouterr().err
    assert script.main(["--input", str(tmp_path / "nope.csv")]) == 1


def test_train_cli_end_to_end_with_flag(demo_clean, tmp_path, capsys):
    csv = tmp_path / "laps.csv"
    demo_clean.to_csv(csv, index=False)
    out = tmp_path / "out"
    code = load_script("train_model").main(
        ["--input", str(csv), "--output-dir", str(out), "--allow-synthetic"]
    )
    assert code == 0
    assert (out / "evaluation.json").exists() and (out / "lap_time_model.joblib").exists()
    assert "SYNTHETIC" in capsys.readouterr().out
