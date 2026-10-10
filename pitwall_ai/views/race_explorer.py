"""Race Explorer page: lap-by-lap analysis of one session."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitwall_ai import charts
from pitwall_ai.config import SUPPORTED_SESSIONS
from pitwall_ai.data.fastf1_loader import DataUnavailableError
from pitwall_ai.views import common
from pitwall_ai.views.common import AppContext


def _historical_session() -> tuple[pd.DataFrame, pd.DataFrame, dict, str] | None:
    """Season / round / session pickers plus a Load button. Returns data once loaded."""
    this_year = pd.Timestamp.now().year
    c1, c2, c3 = st.columns([1, 2, 1])
    season = c1.selectbox("Season", list(range(this_year, 2017, -1)))
    try:
        schedule = common.cached_schedule(int(season))
    except (DataUnavailableError, ValueError) as exc:
        st.error(f"Could not load the schedule. {exc}")
        st.info(
            "Historical mode needs an internet connection the first time. Demo mode works offline."
        )
        return None
    names = [f"{int(r.RoundNumber)} - {r.EventName}" for r in schedule.itertuples()]
    if not names:
        st.warning("No events found for this season.")
        return None
    choice = c2.selectbox("Grand Prix", names)
    session_type = c3.selectbox(
        "Session", list(SUPPORTED_SESSIONS), format_func=lambda k: SUPPORTED_SESSIONS[k]
    )
    round_number = int(choice.split(" - ", 1)[0])
    event_name = choice.split(" - ", 1)[1]

    if st.button("Load session", type="primary"):
        st.session_state["explorer_request"] = (int(season), round_number, session_type, event_name)
    request = st.session_state.get("explorer_request")
    if request is None:
        st.info("Choose a session and press **Load session**.")
        return None
    try:
        all_laps, cleaned, report = common.cached_session(*request)
    except (DataUnavailableError, ValueError) as exc:
        st.error(str(exc))
        return None
    label = f"{request[0]} {request[3]} ({SUPPORTED_SESSIONS[request[2]]})"
    return all_laps, cleaned, report, label


def render(ctx: AppContext) -> None:
    st.subheader("Race Explorer")
    mode = st.radio(
        "Data mode", ["Demo (synthetic)", "Historical (live FastF1 load)"], horizontal=True
    )

    report: dict | None = None
    if mode.startswith("Demo"):
        raw, cleaned = common.demo_frames()
        events = sorted(cleaned["EventName"].unique())
        event = st.selectbox("Demo event", events)
        all_laps = raw[raw["EventName"] == event]
        # Pick the first edition of this circuit for a single-race view.
        first_id = sorted(all_laps["EventId"].unique())[0]
        all_laps = all_laps[all_laps["EventId"] == first_id]
        laps = cleaned[cleaned["EventId"] == first_id]
        label, synthetic = f"{event} [synthetic edition {first_id}]", True
    else:
        loaded = _historical_session()
        if loaded is None:
            return
        all_laps, laps, report, label = loaded
        synthetic = False

    common.data_banner(synthetic)
    st.markdown(f"**{label}**")
    if laps.empty:
        st.warning("No eligible laps remain after cleaning for this session.")
        return

    drivers = sorted(laps["Driver"].dropna().astype(str).unique())
    selected = st.multiselect("Drivers", drivers, default=drivers[:3])

    cols = st.columns(4)
    with cols[0]:
        common.metric_card("Eligible laps", f"{len(laps):,}", f"of {len(all_laps):,} total")
    with cols[1]:
        common.metric_card("Fastest eligible", f"{laps['LapTimeSeconds'].min():.3f} s")
    with cols[2]:
        common.metric_card("Median lap", f"{laps['LapTimeSeconds'].median():.3f} s")
    with cols[3]:
        common.metric_card("Compounds", ", ".join(sorted(laps["Compound"].astype(str).unique())))

    st.plotly_chart(charts.driver_laps_chart(laps, selected), width="stretch")
    left, right = st.columns(2)
    subset = laps[laps["Driver"].isin(selected)] if selected else laps
    left.plotly_chart(
        charts.tyre_age_chart(subset, "Lap time vs tyre age (selection)"), width="stretch"
    )
    right.plotly_chart(
        charts.stint_chart(all_laps[all_laps["Driver"].isin(selected)] if selected else all_laps),
        width="stretch",
    )

    summary = (
        laps.groupby("Driver")
        .agg(
            Laps=("LapNumber", "count"),
            Fastest=("LapTimeSeconds", "min"),
            Median=("LapTimeSeconds", "median"),
            MeanTyreAge=("TyreLife", "mean"),
            Compounds=("Compound", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
        .round(3)
        .reset_index()
        .sort_values("Fastest")
    )
    st.markdown("**Driver summary (eligible laps only)**")
    st.dataframe(summary, width="stretch", hide_index=True)
    if report:
        with st.expander("What the cleaning rules removed"):
            st.json(report["removed"])
