"""Smoke tests: every page of the Streamlit app renders offline in demo mode."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pitwall_ai.views.common import PAGES

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_demo_mode_is_labelled_synthetic():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert any("DEMONSTRATION DATA" in w.value for w in at.warning)
