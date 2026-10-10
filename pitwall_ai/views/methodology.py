"""Methodology page - renders docs/methodology.md."""

from __future__ import annotations

import streamlit as st

from pitwall_ai.config import DOCS_DIR
from pitwall_ai.views.common import AppContext, disclaimer


def render(ctx: AppContext) -> None:
    del ctx
    st.subheader("Methodology")
    path = DOCS_DIR / "methodology.md"
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.error(f"Methodology document not found at {path}.")
    disclaimer()
