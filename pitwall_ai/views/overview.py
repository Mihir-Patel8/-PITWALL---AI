"""Overview page."""

from __future__ import annotations

import streamlit as st

from pitwall_ai import charts
from pitwall_ai.views.common import PAGES, AppContext, data_banner, metric_card


def _go(page: str) -> None:
    st.session_state["page"] = page


def render(ctx: AppContext) -> None:
    st.subheader("Overview")
    st.write(
        "PITWALL AI explores race pace and pit-stop decisions with historical lap data, "
        "machine learning and a strategy simulator. It is an educational project, not "
        "race-engineering software."
    )
    data_banner(ctx.synthetic)
    laps = ctx.laps
    if laps.empty:
        st.info("The selected dataset contains no eligible laps.")
        return

    cols = st.columns(5)
    with cols[0]:
        metric_card("Data mode", ctx.mode_label, ctx.source)
    with cols[1]:
        metric_card("Lap records", f"{len(laps):,}", f"{laps['EventId'].nunique()} events")
    with cols[2]:
        metric_card("Drivers", f"{laps['Driver'].nunique()}")
    with cols[3]:
        metric_card("Median lap", f"{laps['LapTimeSeconds'].median():.2f} s", "all events pooled")
    with cols[4]:
        metric_card("Fastest eligible", f"{laps['LapTimeSeconds'].min():.2f} s")
    if laps["EventId"].nunique() > 1:
        st.caption(
            "Median/fastest pool different circuits, so they only describe the data set - "
            "compare lap times within one event in the Race Explorer."
        )
    compounds = ", ".join(sorted(laps["Compound"].dropna().astype(str).unique()))
    st.write(f"**Tyre compounds present:** {compounds or 'none'}")

    left, right = st.columns(2)
    left.plotly_chart(charts.lap_time_histogram(laps), width="stretch")
    right.plotly_chart(charts.tyre_age_chart(laps), width="stretch")

    st.markdown("**Explore**")
    nav = st.columns(4)
    for column, page in zip(nav, PAGES[1:], strict=True):
        column.button(page, key=f"nav_{page}", on_click=_go, args=(page,), width="stretch")
