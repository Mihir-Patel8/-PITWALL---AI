"""Plotly figure builders. Every function returns a figure and never touches Streamlit.

Empty or unsuitable input yields a figure containing an explanatory message instead of an
exception, so the UI never shows a blank or broken chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pitwall_ai.config import COLORS, COMPOUND_COLORS

_FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": COLORS["text"]}},
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["panel"],
        font={"family": _FONT, "color": COLORS["text"]},
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
        height=height,
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


def empty_figure(message: str, title: str = "") -> go.Figure:
    """A placeholder chart that explains why nothing is shown."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 14, "color": COLORS["muted"]},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title)


def _has(df: pd.DataFrame | None, *columns: str) -> bool:
    return df is not None and not df.empty and all(c in df.columns for c in columns)


def lap_time_histogram(laps: pd.DataFrame, title: str = "Lap-time distribution") -> go.Figure:
    """Distribution of lap time relative to each event's median, split by compound."""
    if not _has(laps, "LapTimeSeconds", "EventId", "Compound"):
        return empty_figure("No lap data available for this chart.", title)
    relative = laps["LapTimeSeconds"] - laps.groupby("EventId")["LapTimeSeconds"].transform(
        "median"
    )
    fig = go.Figure()
    for compound, group in laps.assign(rel=relative).groupby("Compound"):
        fig.add_histogram(
            x=group["rel"],
            name=str(compound),
            opacity=0.75,
            nbinsx=60,
            marker_color=COMPOUND_COLORS.get(str(compound), COLORS["muted"]),
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Lap time relative to event median (s)")
    fig.update_yaxes(title="Laps")
    return _style(fig, title)


def tyre_age_chart(laps: pd.DataFrame, title: str = "Lap time vs tyre age") -> go.Figure:
    """Scatter plus median trend of event-relative lap time against tyre age."""
    if not _has(laps, "LapTimeSeconds", "TyreLife", "Compound", "EventId"):
        return empty_figure("Tyre-age data is not available for this selection.", title)
    data = laps.assign(
        rel=laps["LapTimeSeconds"] - laps.groupby("EventId")["LapTimeSeconds"].transform("median")
    )
    fig = go.Figure()
    for compound, group in data.groupby("Compound"):
        color = COMPOUND_COLORS.get(str(compound), COLORS["muted"])
        sample = group.sample(min(len(group), 1500), random_state=0)
        fig.add_scatter(
            x=sample["TyreLife"],
            y=sample["rel"],
            mode="markers",
            name=f"{compound}",
            marker={"size": 4, "color": color, "opacity": 0.35},
        )
        trend = group.groupby("TyreLife")["rel"].median().reset_index()
        fig.add_scatter(
            x=trend["TyreLife"],
            y=trend["rel"],
            mode="lines",
            name=f"{compound} median",
            line={"color": color, "width": 3},
        )
    fig.update_xaxes(title="Tyre age (laps)")
    fig.update_yaxes(title="Lap time relative to event median (s)")
    return _style(fig, title)


def driver_laps_chart(
    laps: pd.DataFrame, drivers: list[str], title: str = "Lap times by driver"
) -> go.Figure:
    """Lap time against lap number for the selected drivers."""
    if not _has(laps, "Driver", "LapNumber", "LapTimeSeconds") or not drivers:
        return empty_figure("Select at least one driver with lap data.", title)
    fig = go.Figure()
    for driver in drivers:
        group = laps[laps["Driver"] == driver].sort_values("LapNumber")
        fig.add_scatter(
            x=group["LapNumber"],
            y=group["LapTimeSeconds"],
            mode="lines+markers",
            name=str(driver),
            marker={"size": 4},
        )
    fig.update_xaxes(title="Lap")
    fig.update_yaxes(title="Lap time (s)")
    return _style(fig, title)


def stint_chart(laps: pd.DataFrame, title: str = "Tyre stints") -> go.Figure:
    """Horizontal bars showing each driver's stints coloured by compound."""
    if not _has(laps, "Driver", "Stint", "Compound", "LapNumber"):
        return empty_figure("Stint information is not available for this session.", title)
    data = laps.dropna(subset=["Stint", "Compound"])
    if data.empty:
        return empty_figure("Stint information is not available for this session.", title)
    stints = (
        data.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
        .agg(start="min", end="max")
        .reset_index()
    )
    fig = go.Figure()
    for compound, group in stints.groupby("Compound"):
        fig.add_bar(
            y=group["Driver"],
            x=group["end"] - group["start"] + 1,
            base=group["start"] - 1,
            orientation="h",
            name=str(compound),
            marker={
                "color": COMPOUND_COLORS.get(str(compound), COLORS["muted"]),
                "line": {"color": COLORS["background"], "width": 1},
            },
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Lap")
    fig.update_yaxes(autorange="reversed", categoryorder="category ascending")
    return _style(fig, title, height=max(320, 22 * stints["Driver"].nunique() + 120))


def strategy_delta_chart(ranking: pd.DataFrame, top_n: int = 12, title: str = "") -> go.Figure:
    """Time lost relative to the best candidate (lower is better)."""
    title = title or "Estimated time behind best candidate"
    if not _has(ranking, "Strategy", "DeltaToBest", "Stops"):
        return empty_figure("No strategies to display.", title)
    top = ranking.head(top_n).iloc[::-1]
    colors = [
        COLORS["red"] if s == 1 else COLORS["cyan"] if s == 2 else COLORS["teal"]
        for s in top["Stops"]
    ]
    fig = go.Figure(
        go.Bar(x=top["DeltaToBest"], y=top["Strategy"], orientation="h", marker_color=colors)
    )
    fig.update_xaxes(title="Seconds behind best candidate (estimated)")
    return _style(fig, title, height=max(320, 30 * len(top) + 120))


def strategy_lap_chart(
    results: dict[str, np.ndarray], title: str = "Estimated lap times by strategy"
) -> go.Figure:
    """Estimated lap-time profile of several strategies (pit-lane time excluded)."""
    if not results:
        return empty_figure("No strategies to display.", title)
    fig = go.Figure()
    for name, times in results.items():
        fig.add_scatter(x=np.arange(1, len(times) + 1), y=times, mode="lines", name=name)
    fig.update_xaxes(title="Remaining lap")
    fig.update_yaxes(title="Estimated lap time (s)")
    return _style(fig, title)


def monte_carlo_chart(
    summary: pd.DataFrame, title: str = "Simulated spread under uncertain assumptions"
) -> go.Figure:
    """Mean with P10-P90 range of simulated outcomes (relative to the best mean)."""
    if not _has(summary, "Strategy", "MeanTime", "P10", "P90"):
        return empty_figure("Run the Monte Carlo simulation to see this chart.", title)
    best = summary["MeanTime"].min()
    fig = go.Figure(
        go.Scatter(
            x=summary["MeanTime"] - best,
            y=summary["Strategy"],
            mode="markers",
            marker={"color": COLORS["cyan"], "size": 10},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": summary["P90"] - summary["MeanTime"],
                "arrayminus": summary["MeanTime"] - summary["P10"],
                "color": COLORS["muted"],
            },
        )
    )
    fig.update_xaxes(title="Mean simulated time vs best mean (s), bars = P10-P90 of simulated runs")
    return _style(fig, title, height=max(300, 60 * len(summary) + 120))


def model_comparison_chart(
    metrics: dict[str, dict[str, float | None]], title: str = "Held-out error by model"
) -> go.Figure:
    """Grouped bars of MAE and RMSE for every evaluated model."""
    if not metrics:
        return empty_figure("No evaluation results available.", title)
    names = list(metrics)
    fig = go.Figure()
    fig.add_bar(
        x=names, y=[metrics[n]["mae"] for n in names], name="MAE (s)", marker_color=COLORS["red"]
    )
    fig.add_bar(
        x=names, y=[metrics[n]["rmse"] for n in names], name="RMSE (s)", marker_color=COLORS["cyan"]
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Seconds (lower is better)")
    return _style(fig, title)


def predicted_vs_observed_chart(
    predictions: pd.DataFrame, model: str, title: str = ""
) -> go.Figure:
    """Held-out predictions against observed lap times with the y = x reference line."""
    column = f"pred_{model}"
    title = title or f"Predicted vs observed ({model}, held-out events)"
    if not _has(predictions, "observed", column):
        return empty_figure("No held-out predictions available.", title)
    lo, hi = predictions["observed"].min(), predictions["observed"].max()
    fig = go.Figure()
    fig.add_scatter(
        x=predictions["observed"],
        y=predictions[column],
        mode="markers",
        marker={"size": 4, "color": COLORS["cyan"], "opacity": 0.4},
        name="laps",
    )
    fig.add_scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        name="perfect prediction",
        line={"color": COLORS["red"], "dash": "dash"},
    )
    fig.update_xaxes(title="Observed lap time (s)")
    fig.update_yaxes(title="Predicted lap time (s)")
    return _style(fig, title)


def residual_chart(predictions: pd.DataFrame, model: str, title: str = "") -> go.Figure:
    """Histogram of residuals (observed - predicted)."""
    column = f"pred_{model}"
    title = title or f"Residuals ({model}, held-out events)"
    if not _has(predictions, "observed", column):
        return empty_figure("No held-out predictions available.", title)
    residual = predictions["observed"] - predictions[column]
    fig = go.Figure(go.Histogram(x=residual, nbinsx=60, marker_color=COLORS["teal"]))
    fig.update_xaxes(title="Observed - predicted (s)")
    fig.update_yaxes(title="Laps")
    return _style(fig, title)


def importance_chart(
    importance: pd.DataFrame, title: str = "Permutation importance (held-out events)"
) -> go.Figure:
    """Bar chart of the MAE increase when each input is shuffled."""
    if not _has(importance, "feature", "mae_increase_seconds"):
        return empty_figure("Permutation importance is not available.", title)
    data = importance.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["mae_increase_seconds"],
            y=data["feature"],
            orientation="h",
            error_x={"type": "data", "array": data.get("std"), "color": COLORS["muted"]},
            marker_color=COLORS["red"],
        )
    )
    fig.update_xaxes(title="Increase in MAE when shuffled (s)")
    return _style(fig, title, height=320)
