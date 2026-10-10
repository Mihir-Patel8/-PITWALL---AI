"""Strategy Lab page: compare candidate pit-stop strategies."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitwall_ai import charts
from pitwall_ai.config import (
    DEFAULT_MONTE_CARLO_RUNS,
    DEFAULT_PIT_LOSS_SECONDS,
    DRY_COMPOUNDS,
    MODEL_PATH,
)
from pitwall_ai.data.demo_data import DEMO_ASSUMPTIONS
from pitwall_ai.ml.prediction import ModelNotFoundError, load_predictor
from pitwall_ai.strategy import optimizer, simulator
from pitwall_ai.strategy.estimators import (
    EmpiricalEstimator,
    ModelLapEstimator,
    ParametricEstimator,
)
from pitwall_ai.strategy.scenarios import Scenario
from pitwall_ai.views.common import AppContext, metric_card


def _manual_estimator() -> ParametricEstimator:
    st.markdown("**Pace and tyre assumptions** (defaults are the synthetic demonstration values)")
    cols = st.columns(4)
    base = cols[0].number_input("Base lap time, medium tyre, lap 1 (s)", 50.0, 150.0, 90.0, 0.5)
    fuel = cols[1].number_input("Fuel-burn gain (s / lap)", 0.0, 0.1, 0.03, 0.005, format="%.3f")
    offsets, degradation = {}, {}
    for compound in DRY_COMPOUNDS:
        c1, c2 = st.columns(2)
        offsets[compound] = c1.number_input(
            f"{compound}: pace offset (s)",
            -3.0,
            3.0,
            DEMO_ASSUMPTIONS["compound_offset"][compound],
            0.1,
            key=f"off_{compound}",
        )
        degradation[compound] = c2.number_input(
            f"{compound}: degradation (s / lap)",
            0.0,
            0.5,
            DEMO_ASSUMPTIONS["degradation_per_lap"][compound],
            0.005,
            format="%.3f",
            key=f"deg_{compound}",
        )
    return ParametricEstimator(base, offsets, degradation, fuel, name="Manual assumptions")


def _choose_estimator(ctx: AppContext):
    options = ["Manual assumptions"]
    if len(ctx.laps) >= 30:
        options.append("Empirical fit to selected dataset")
    if MODEL_PATH.exists():
        options.append("Trained model")
    source = st.radio("Lap-time estimator", options, horizontal=True)
    if source == "Manual assumptions":
        return _manual_estimator(), True
    if source.startswith("Empirical"):
        base = st.number_input(
            "Reference pace: event-median lap time (s)",
            50.0,
            150.0,
            round(float(ctx.laps["LapTimeSeconds"].median()), 1),
            0.5,
        )
        try:
            estimator = EmpiricalEstimator.fit(ctx.laps, base)
        except ValueError as exc:
            st.error(f"Could not fit the empirical estimator: {exc}")
            return None, ctx.synthetic
        return estimator, ctx.synthetic
    try:
        predictor = load_predictor()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return None, False
    c1, c2 = st.columns(2)
    driver = c1.selectbox("Driver", sorted(ctx.laps["Driver"].astype(str).unique()))
    event = c2.selectbox("Circuit / event", sorted(ctx.laps["EventName"].astype(str).unique()))
    synthetic = bool(predictor.report.get("is_synthetic"))
    if synthetic:
        st.warning("This model was trained on SYNTHETIC demo data.")
    return ModelLapEstimator(predictor, driver, event), synthetic


def render(ctx: AppContext) -> None:
    st.subheader("Strategy Lab")
    st.write(
        "Compare pit-stop plans by their **estimated remaining race time under the assumptions "
        "below**. Results are model estimates, not predictions of a real race."
    )

    with st.expander("Race set-up", expanded=True):
        c = st.columns(4)
        total_laps = c[0].number_input("Remaining laps", 10, 78, 56)
        start_lap = c[1].number_input("Current race lap", 1, 78, 1)
        start_age = c[2].number_input("Current tyre age (laps)", 1, 40, 1)
        pit_loss = c[3].number_input(
            "Pit-stop time loss (s)", 5.0, 40.0, DEFAULT_PIT_LOSS_SECONDS, 0.5
        )
        c = st.columns(4)
        start_compound = c[0].selectbox("Starting compound", DRY_COMPOUNDS, index=1)
        available = c[1].multiselect(
            "Compounds available for later stints", DRY_COMPOUNDS, DRY_COMPOUNDS
        )
        max_stops = c[2].selectbox("Max pit stops", [1, 2, 3], index=1)
        min_stops = c[3].selectbox("Min pit stops", [0, 1, 2], index=1)
        c = st.columns(3)
        min_stint = c[0].number_input("Min stint length (laps)", 3, 40, 8)
        step = c[1].number_input(
            "Pit-lap step (laps)",
            1,
            10,
            1,
            help="Larger steps test fewer pit-lap options and run faster.",
        )
        two_compounds = c[2].checkbox("Require two different compounds", True)

    with st.expander("Optional hypothetical scenario (not a prediction)"):
        c = st.columns(3)
        deg_scale = c[0].slider("Tyre degradation multiplier", 0.5, 2.0, 1.0, 0.05)
        pace_offset = c[1].slider("All laps slower by (s)", 0.0, 10.0, 0.0, 0.5)
        sc_on = c[2].checkbox("Include a hypothetical Safety Car")
        sc_start, sc_len = 1, 0
        if sc_on:
            c = st.columns(2)
            sc_start = int(c[0].number_input("Safety Car first lap", 1, 120, 20))
            sc_len = int(c[1].number_input("Safety Car laps", 1, 15, 4))
        scenario = Scenario(
            "User scenario",
            deg_scale,
            pace_offset,
            sc_start if sc_on else None,
            sc_len if sc_on else 0,
        )

    estimator, synthetic = _choose_estimator(ctx)
    if estimator is None:
        return
    if synthetic or isinstance(estimator, ParametricEstimator):
        st.caption(
            "Lap-time inputs are illustrative assumptions unless you entered values you trust."
        )

    if start_compound not in available and available:
        available = [*available, start_compound]
    try:
        config = simulator.SimulationConfig(
            int(total_laps), float(pit_loss), int(start_lap), int(start_age), scenario
        )
        config.validate()
        candidates = optimizer.generate_candidates(
            int(total_laps),
            start_compound,
            tuple(dict.fromkeys(available or [start_compound])),
            min_stops=int(min_stops),
            max_stops=int(max(max_stops, min_stops)),
            min_stint=int(min_stint),
            step=int(step),
            require_two_compounds=two_compounds,
        )
        ranking = optimizer.rank_strategies(candidates, estimator, config)
    except ValueError as exc:
        st.error(f"Cannot evaluate these settings: {exc}")
        return

    best = ranking.iloc[0]
    cols = st.columns(4)
    with cols[0]:
        metric_card("Best candidate", best["Strategy"])
    with cols[1]:
        metric_card(
            "Estimated remaining time",
            f"{best['TotalTime'] / 60:.2f} min",
            f"{best['TotalTime']:.1f} s",
        )
    with cols[2]:
        metric_card(
            "Estimated pit cost", f"{best['PitCost']:.1f} s", f"{int(best['Stops'])} stop(s)"
        )
    with cols[3]:
        metric_card("Candidates evaluated", f"{len(ranking):,}")

    st.info(optimizer.explain_choice(ranking))
    left, right = st.columns(2)
    left.plotly_chart(charts.strategy_delta_chart(ranking), width="stretch")
    picks = {}
    for _, row in (
        pd.concat([ranking.head(2), optimizer.best_by_stop_count(ranking)])
        .drop_duplicates("Strategy")
        .iterrows()
    ):
        strategy = next(s for s in candidates if s.label() == row["Strategy"])
        picks[row["Strategy"]] = simulator.simulate_strategy(strategy, estimator, config).lap_times
    right.plotly_chart(charts.strategy_lap_chart(picks), width="stretch")

    st.markdown("**Strategy ranking (top 15)**")
    show = ranking.head(15).copy()
    for col in ("DrivingTime", "PitCost", "TotalTime", "DeltaToBest"):
        show[col] = show[col].round(2)
    st.dataframe(show, width="stretch", hide_index=True)
    st.markdown("**Best one-stop vs best two-stop (and others)**")
    st.dataframe(optimizer.best_by_stop_count(ranking).round(2), width="stretch", hide_index=True)

    with st.expander("Monte Carlo: how much does the ranking depend on uncertain assumptions?"):
        c = st.columns(4)
        runs = int(c[0].number_input("Runs", 50, 5000, DEFAULT_MONTE_CARLO_RUNS, 50))
        seed = int(c[1].number_input("Random seed", 0, 10**6, 42))
        deg_sigma = c[2].slider("Degradation uncertainty (lognormal sigma)", 0.0, 0.6, 0.2, 0.05)
        pit_sigma = c[3].slider("Pit-loss std dev (s)", 0.0, 4.0, 1.0, 0.25)
        st.caption(
            "Each run perturbs the degradation rate and pace offset of each compound and the "
            "loss of each pit stop. The P10-P90 range describes simulated outcomes under these "
            "assumed distributions; it is NOT a statistical confidence interval."
        )
        top = [
            next(s for s in candidates if s.label() == name) for name in ranking.head(5)["Strategy"]
        ]
        mc = simulator.monte_carlo(
            top,
            estimator,
            config,
            n_runs=runs,
            degradation_sigma=deg_sigma,
            pit_loss_sigma=pit_sigma,
            seed=seed,
        )
        st.plotly_chart(charts.monte_carlo_chart(mc), width="stretch")
        st.dataframe(mc.round(3), width="stretch", hide_index=True)

    with st.expander("Assumptions used"):
        lines = [
            f"- Estimator: **{estimator.name}**",
            f"- Remaining laps: {total_laps} (from lap {start_lap}); tyre age {start_age}",
            f"- Pit-stop loss: {pit_loss:.1f} s per stop, counted once per stop",
            "- Tyre age resets to 1 after each stop",
            "- Not modelled: traffic, fuel-load limits, tyre-set availability, risk",
            *[f"- Scenario: {line}" for line in scenario.describe()],
        ]
        st.markdown("\n".join(lines))
