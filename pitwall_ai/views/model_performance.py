"""Model Performance page - shows only genuine, saved evaluation results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitwall_ai import charts
from pitwall_ai.config import TEST_PREDICTIONS_PATH
from pitwall_ai.ml.prediction import load_evaluation_report
from pitwall_ai.views.common import AppContext, metric_card


@st.cache_data
def _load_predictions(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


def render(ctx: AppContext) -> None:
    del ctx  # evaluation results come from saved artifacts, not from the selected data
    st.subheader("Model Performance")
    report = load_evaluation_report()
    if report is None:
        st.info(
            "No trained model or evaluation report was found, so no metrics are shown (nothing is "
            "invented). To create them:\n\n"
            "```\npython scripts/collect_data.py --start-year 2023 --end-year 2024 --max-events 8\n"
            "python scripts/train_model.py --input data/processed/laps.csv\n```\n"
            "The strategy simulator and demo mode work without a model."
        )
        return

    if report.get("is_synthetic"):
        st.warning(
            "**These results come from SYNTHETIC demonstration data.** They show the pipeline "
            "works, not how well it predicts real F1 laps."
        )
    else:
        st.success("Evaluation on historical data collected with FastF1.")

    cols = st.columns(4)
    with cols[0]:
        metric_card(
            "Events (train / test)", f"{report['n_events_train']} / {report['n_events_test']}"
        )
    with cols[1]:
        metric_card(
            "Laps (train / test)", f"{report['n_rows_train']:,} / {report['n_rows_test']:,}"
        )
    with cols[2]:
        metric_card("Best eligible model", str(report.get("best_model") or "none"))
    with cols[3]:
        metric_card("Created (UTC)", str(report["created_utc"])[:10])
    st.caption(
        f"Split: {report['split_method']}. Held-out events: {', '.join(report['test_event_ids'])}"
    )

    metrics = report["metrics"]
    table = pd.DataFrame(metrics).T.rename(
        columns={"mae": "MAE (s)", "rmse": "RMSE (s)", "r2": "R²"}
    )
    st.dataframe(table.round(3), width="stretch")
    st.plotly_chart(charts.model_comparison_chart(metrics), width="stretch")

    best = report.get("best_model")
    if best is None:
        st.error(
            "No model beat the baseline on held-out events, so no model was saved. "
            "This is a genuine result: more or different data may be needed."
        )
    elif TEST_PREDICTIONS_PATH.exists():
        predictions = _load_predictions(
            str(TEST_PREDICTIONS_PATH), TEST_PREDICTIONS_PATH.stat().st_mtime
        )
        models = [c.removeprefix("pred_") for c in predictions.columns if c.startswith("pred_")]
        model = st.selectbox(
            "Model to inspect", models, index=models.index(best) if best in models else 0
        )
        left, right = st.columns(2)
        left.plotly_chart(charts.predicted_vs_observed_chart(predictions, model), width="stretch")
        right.plotly_chart(charts.residual_chart(predictions, model), width="stretch")
        st.caption("Predictions shown are a sample (max 5,000 laps) of the held-out events.")

    importance = pd.DataFrame(report.get("importance", []))
    if not importance.empty:
        st.plotly_chart(charts.importance_chart(importance), width="stretch")
        st.caption(str(report.get("importance_note", "")))

    with st.expander("Limitations of this evaluation"):
        st.markdown(
            "- A single random event split; results can change with another seed or more events.\n"
            "- Lap time depends heavily on circuit, traffic, weather and car, most of which are "
            "not inputs. Errors of a few tenths of a second are large in racing terms.\n"
            "- Circuits unseen in training cannot be modelled by circuit-specific pace.\n"
            "- Permutation importance is indicative only when inputs are correlated."
        )
