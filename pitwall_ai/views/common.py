"""Shared UI helpers: app context, cached data loaders, banners and metric cards."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from pitwall_ai.config import COLORS, DEFAULT_DATASET_PATH, DISCLAIMER
from pitwall_ai.data import fastf1_loader
from pitwall_ai.data.cleaning import clean_laps
from pitwall_ai.data.demo_data import DEMO_LABEL, generate_demo_laps

PAGES = ["Overview", "Race Explorer", "Strategy Lab", "Model Performance", "Methodology"]
SOURCE_DEMO = "Demo data (synthetic)"
SOURCE_DATASET = "Collected historical dataset"


@dataclass
class AppContext:
    """What the pages need to know about the currently selected data."""

    laps: pd.DataFrame
    source: str
    synthetic: bool

    @property
    def mode_label(self) -> str:
        return "DEMO" if self.synthetic else "HISTORICAL DATA"


@st.cache_data(show_spinner="Generating synthetic demo data ...")
def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw_demo_laps, cleaned_demo_laps)`` - deterministic and offline."""
    raw = generate_demo_laps()
    cleaned, _ = clean_laps(raw)
    return raw, cleaned


@st.cache_data(show_spinner="Reading dataset ...")
def read_dataset(path: str, modified_time: float) -> pd.DataFrame:
    """Read a processed CSV; ``modified_time`` invalidates the cache when the file changes."""
    del modified_time  # only part of the cache key
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading season schedule from FastF1 ...", ttl=3600)
def cached_schedule(season: int) -> pd.DataFrame:
    return fastf1_loader.get_event_schedule(season)


@st.cache_data(show_spinner="Loading session from FastF1 (first download can take a minute) ...")
def cached_session(season: int, round_number: int, session_type: str, event_name: str):
    """Return ``(all_laps, cleaned_laps, cleaning_report)`` for one historical session."""
    all_laps, _ = fastf1_loader.load_session_laps(
        season, round_number, session_type, event_name=event_name, clean=False
    )
    cleaned, report = clean_laps(all_laps)
    return all_laps, cleaned, report.to_dict()


def dataset_available() -> bool:
    return DEFAULT_DATASET_PATH.exists()


def data_banner(synthetic: bool) -> None:
    """Prominent label that must accompany any synthetic data."""
    if synthetic:
        st.warning(f"**{DEMO_LABEL}.** Charts and numbers here illustrate the software only.")
    else:
        st.success(
            "**HISTORICAL DATA** collected with FastF1 (see Methodology for cleaning rules)."
        )


def metric_card(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""<div class="pw-card"><div class="pw-label">{label}</div>
        <div class="pw-value">{value}</div><div class="pw-hint">{hint}</div></div>""",
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        f"""<style>
        .pw-title {{font-size: 2.1rem; font-weight: 800; letter-spacing: .08em;}}
        .pw-title span {{color: {COLORS["red"]};}}
        .pw-sub {{color: {COLORS["muted"]}; margin-top: -.4rem; margin-bottom: .8rem;}}
        .pw-card {{background: {COLORS["panel"]}; border-left: 4px solid {COLORS["red"]};
                  padding: .7rem 1rem; border-radius: 6px; margin-bottom: .6rem;}}
        .pw-label {{color: {COLORS["muted"]}; font-size: .75rem; text-transform: uppercase;
                   letter-spacing: .08em;}}
        .pw-value {{font-size: 1.5rem; font-weight: 700; color: {COLORS["text"]};}}
        .pw-hint {{color: {COLORS["cyan"]}; font-size: .75rem;}}
        </style>""",
        unsafe_allow_html=True,
    )


def header() -> None:
    st.markdown('<div class="pw-title">PITWALL<span> AI</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pw-sub">Formula 1 Race Strategy Intelligence System</div>',
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.caption(DISCLAIMER)
