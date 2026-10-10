"""PITWALL AI - Streamlit entry point.  Run with:  streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from pitwall_ai.config import DEFAULT_DATASET_PATH
from pitwall_ai.data.cleaning import DataValidationError, validate_columns
from pitwall_ai.data.demo_data import is_synthetic
from pitwall_ai.views import (
    common,
    methodology,
    model_performance,
    overview,
    race_explorer,
    strategy_lab,
)

st.set_page_config(page_title="PITWALL AI", page_icon="🏁", layout="wide")
common.inject_css()


def _select_context() -> common.AppContext:
    """Sidebar: choose the dataset used by Overview and Strategy Lab."""
    options = [common.SOURCE_DEMO]
    if common.dataset_available():
        options.insert(0, common.SOURCE_DATASET)
    source = st.sidebar.radio("Data source", options)
    if source == common.SOURCE_DATASET:
        try:
            laps = common.read_dataset(
                str(DEFAULT_DATASET_PATH), DEFAULT_DATASET_PATH.stat().st_mtime
            )
            validate_columns(laps)
            return common.AppContext(laps, f"{DEFAULT_DATASET_PATH.name}", is_synthetic(laps))
        except (DataValidationError, OSError, ValueError) as exc:
            st.sidebar.error(
                f"Could not use the collected dataset: {exc}. Falling back to demo data."
            )
    _, cleaned = common.demo_frames()
    if not common.dataset_available():
        st.sidebar.info("No collected dataset found - showing synthetic demo data.")
    return common.AppContext(cleaned, "synthetic generator", True)


def main() -> None:
    common.header()
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Section", common.PAGES, key="page", label_visibility="collapsed")
    ctx = _select_context()
    st.sidebar.caption(
        "Independent educational project - not affiliated with Formula One, the FIA, "
        "any team or driver."
    )
    {
        "Overview": overview.render,
        "Race Explorer": race_explorer.render,
        "Strategy Lab": strategy_lab.render,
        "Model Performance": model_performance.render,
        "Methodology": methodology.render,
    }[page](ctx)
    if page != "Methodology":
        common.disclaimer()


main()
